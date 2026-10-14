"""Annotation utilities for RiboNet.

Provides an ``AnnotationDB`` class that wraps a single gffutils in-memory
database built from a GFF3 or GTF annotation file.  Building the database is
expensive, so it is done once in ``__init__`` and reused across all methods.

Supported input formats (detected automatically from the file extension):
    * GFF3  (.gff, .gff3)
    * GTF   (.gtf)

Typical usage::

    db = AnnotationDB("/path/to/annotation.gtf")
    cds_info   = db.parse_cds_starts()
    tx_to_gene = db.build_feature_to_gene_map()
    cds_df     = db.parse_annotation("CDS")
    db.write_aip_tab("/path/to/output.tab")

"""

from __future__ import annotations

import logging
import os

import gffutils
import pandas as pd

log = logging.getLogger(__name__)

_GENE_FEATURETYPES = ["gene", "transposable_element_gene", "blocked_reading_frame"]
_TX_FEATURETYPES = ["mRNA", "transcript"]


# ---------------------------------------------------------------------------
# Module-level pure helper (no database access)
# ---------------------------------------------------------------------------

def _gene_name_from_feature(feat: gffutils.Feature) -> str | None:
    """Return the best available gene name from a gffutils feature.

    Priority: gene_name > gene_id > Name > ID > locus_tag
    Falls back gracefully across GFF3 and GTF conventions.
    """
    for attr in ("gene_name", "gene_id", "Name", "ID", "locus_tag"):
        val = feat.attributes.get(attr)
        if val:
            return val[0]
    return None


# ---------------------------------------------------------------------------
# AnnotationDB class
# ---------------------------------------------------------------------------

class AnnotationDB:
    """Wrapper around a gffutils in-memory database for a single annotation file.

    The database is built once on instantiation; all public methods reuse it.

    Parameters
    ----------
    annotation_file:
        Path to a GFF3 (.gff / .gff3) or GTF (.gtf) annotation file.
    """

    def __init__(self, annotation_file: str) -> None:
        self.annotation_file = annotation_file
        ext = os.path.splitext(annotation_file)[1].lower()
        self._is_gtf = ext == ".gtf"
        log.info("Building annotation database from %s", annotation_file)
        self.db = self._build_db()

    # ------------------------------------------------------------------
    # Pickle support: sqlite3.Connection is not picklable, so drop the
    # database handle on serialisation.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("db", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Internal DB construction
    # ------------------------------------------------------------------

    def _build_db(self) -> gffutils.FeatureDB:
        """Create an in-memory gffutils database from the annotation file."""
        db = gffutils.create_db(
            self.annotation_file,
            dbfn=":memory:",
            force=True,
            keep_order=True,
            merge_strategy="merge",
            sort_attribute_values=True,
            disable_infer_genes=not self._is_gtf,
            disable_infer_transcripts=not self._is_gtf,
        )
        self._merge_stop_codons_into_cds(db)
        return db

    def _merge_stop_codons_into_cds(self, db: gffutils.FeatureDB) -> None:
        """Extend terminal CDS features in *db* to absorb ``stop_codon`` features.

        GENCODE/Ensembl GTF files annotate the stop codon as a separate feature
        type and exclude it from ``CDS`` lines.  This method finds every
        ``stop_codon`` record and extends the adjacent terminal CDS feature so
        that *all* coordinate queries on the database automatically include the
        stop codon, regardless of which public method is called.

        This is a no-op when no ``stop_codon`` features are present (e.g., SGD
        GFF3 files where the CDS already spans the stop codon).
        """
        try:
            next(db.features_of_type("stop_codon"))
        except StopIteration:
            log.debug("No stop_codon features in database; CDS boundaries kept as-is.")
            return

        # Collect all (feature_id, new_start, new_end) updates before touching
        # the DB.  Calling db.update() inside the features_of_type() iterator
        # causes a "database table is locked" OperationalError because gffutils
        # tries to rebuild indexes while the read cursor is still open.
        updates: dict[str, tuple[int, int]] = {}  # feature_id -> (new_start, new_end)

        for stop_feat in list(db.features_of_type("stop_codon")):
            strand = stop_feat.strand
            parents = list(db.parents(stop_feat, featuretype=_TX_FEATURETYPES))
            if not parents:
                parents = list(db.parents(stop_feat, featuretype=_GENE_FEATURETYPES))
            for parent in parents:
                cds_feats = db.children(parent, featuretype="CDS")
                if not cds_feats:
                    continue
                # Only absorb stop codon pieces adjacent to the terminal CDS: a stop codon
                # split by an intron must not extend the CDS over the intron.
                if strand == "+":
                    cds_feats = sorted(cds_feats, key=lambda f: f.end)
                    terminal = cds_feats[-1]
                    if stop_feat.end > terminal.end and stop_feat.start <= terminal.end + 1:
                        cur_start, cur_end = updates.get(terminal.id, (terminal.start, terminal.end))
                        updates[terminal.id] = (cur_start, max(cur_end, stop_feat.end))
                else:
                    cds_feats = sorted(cds_feats, key=lambda f: f.start)
                    terminal = cds_feats[0]
                    if stop_feat.start < terminal.start and stop_feat.end >= terminal.start - 1:
                        cur_start, cur_end = updates.get(terminal.id, (terminal.start, terminal.end))
                        updates[terminal.id] = (min(cur_start, stop_feat.start), cur_end)

        if updates:
            n_extended = 0
            for feat_id, (new_start, new_end) in updates.items():
                db.conn.execute(
                    "UPDATE features SET start = ?, end = ? WHERE id = ?",
                    (new_start, new_end, feat_id),
                )
                n_extended += 1
            db.conn.commit()
            log.info("Extended %d CDS feature(s) to include stop codon.", n_extended)
        else:
            log.debug("No CDS features needed extension to include stop codon.")

    # ------------------------------------------------------------------
    # Internal feature grouping helpers
    # ------------------------------------------------------------------

    def _cds_features_by_gene(self) -> dict[str, list[gffutils.Feature]]:
        """Group CDS features by gene name.

        Returns
        -------
        dict
            gene_name -> sorted list of CDS Feature objects.
        """
        gene_cds: dict[str, list] = {}

        for feat in self.db.features_of_type("CDS"):
            name = None
            parents = list(self.db.parents(feat, featuretype=_GENE_FEATURETYPES))
            for parent in parents:
                name = _gene_name_from_feature(parent)
                if name:
                    break

            if name is None:
                parents = list(self.db.parents(feat, featuretype=_TX_FEATURETYPES))
                for parent in parents:
                    name = _gene_name_from_feature(parent)
                    if name:
                        break

            if name is None:
                log.warning(
                    "Could not determine gene name for CDS at %s:%d-%d; skipping.",
                    feat.seqid, feat.start, feat.end,
                )
                continue

            gene_cds.setdefault(name, []).append(feat)

        for name in gene_cds:
            gene_cds[name].sort(key=lambda f: f.start)

        return gene_cds

    def _codgene_features_by_gene(self) -> dict[str, gffutils.Feature]:
        """Map each gene name to its representative gene/transcript Feature.

        Returns
        -------
        dict
            gene_name -> parent gene/transcript Feature.
        """
        gene_cds: dict[str, gffutils.Feature] = {}

        for feat in self.db.features_of_type("CDS"):
            name = None
            gene_feat = None

            parents = list(self.db.parents(feat, featuretype=_GENE_FEATURETYPES))
            for parent in parents:
                name = _gene_name_from_feature(parent)
                gene_feat = parent
                if name:
                    break

            if name is None:
                parents = list(self.db.parents(feat, featuretype=_TX_FEATURETYPES))
                for parent in parents:
                    name = _gene_name_from_feature(parent)
                    gene_feat = parent
                    if name:
                        break

            if name is None:
                log.warning(
                    "Could not determine gene name for CDS at %s:%d-%d; skipping.",
                    feat.seqid, feat.start, feat.end,
                )
                continue

            gene_cds[name] = gene_feat

        return gene_cds

    # ------------------------------------------------------------------
    # Internal region row builders
    # ------------------------------------------------------------------

    def _cds_row(self, gene_feat, gene_name: str, seqid: str, strand: str) -> dict:
        """Return a row dict for the CDS region of a gene."""
        cds_feats = list(self.db.children(gene_feat, featuretype="CDS", order_by="start"))
        if not cds_feats:
            raise ValueError("no CDS children for gene %s" % gene_name)

        if strand == "+":
            start = min(f.start - 1 for f in cds_feats)
            end = max(f.stop for f in cds_feats)
        else:
            start = max(f.stop for f in cds_feats)
            end = min(f.start - 1 for f in cds_feats)
        length = abs(end - start)
        return dict(gene=gene_name, chr=seqid, start=start, end=end, strand=strand, length=length)

    def _transcript_bounds(self, gene_feat) -> tuple[int, int]:
        """Return (tx_start, tx_end) in gffutils coordinates (0-based half-open)."""
        tx_feats = list(self.db.children(gene_feat, featuretype=_TX_FEATURETYPES, order_by="start"))
        if tx_feats:
            return min(f.start - 1 for f in tx_feats), max(f.stop for f in tx_feats)
        return gene_feat.start - 1, gene_feat.stop

    def _utr5_row(self, gene_feat, gene_name: str, seqid: str, strand: str) -> dict | None:
        """Return a row dict for the 5'UTR of a gene."""
        cds_feats = list(self.db.children(gene_feat, featuretype="CDS", order_by="start"))
        if not cds_feats:
            raise ValueError("no CDS children for gene %s" % gene_name)
        tx_start, tx_end = self._transcript_bounds(gene_feat)

        if strand == "+":
            start = tx_start
            end = min(f.start - 1 for f in cds_feats)
        else:
            start = max(f.stop for f in cds_feats)
            end = tx_end
        length = end - start
        if length <= 0:
            return None
        return dict(gene=gene_name, chr=seqid, start=start, end=end, strand=strand, length=length)

    def _utr3_row(self, gene_feat, gene_name: str, seqid: str, strand: str) -> dict | None:
        """Return a row dict for the 3'UTR of a gene."""
        cds_feats = list(self.db.children(gene_feat, featuretype="CDS", order_by="start"))
        if not cds_feats:
            raise ValueError("no CDS children for gene %s" % gene_name)
        tx_start, tx_end = self._transcript_bounds(gene_feat)

        if strand == "+":
            start = max(f.stop for f in cds_feats)
            end = tx_end
        else:
            start = tx_start
            end = min(f.start - 1 for f in cds_feats)
        length = end - start
        if length <= 0:
            return None
        return dict(gene=gene_name, chr=seqid, start=start, end=end, strand=strand, length=length)

    def _has_utr_intron_features(self) -> bool:
        """Return True if 'five_prime_UTR_intron' features exist in the database (yeast GFF style)."""
        try:
            next(self.db.features_of_type("five_prime_UTR_intron"))
            return True
        except StopIteration:
            return False

    def _exons_from_exon_features(
        self, gene_feat: gffutils.Feature, utr_row: dict | None
    ) -> list[tuple[int, int]]:
        """Derive exonic intervals within a UTR from explicit 'exon' features.

        Exon features are clipped to the UTR boundaries and returned as a sorted
        list of 0-based half-open genomic intervals ``(start, end)``.

        Parameters
        ----------
        gene_feat:
            Representative gene/transcript feature (used to fetch children).
        utr_row:
            Row dict returned by :meth:`_utr5_row` / :meth:`_utr3_row`, or
            ``None`` when no UTR is annotated.

        Returns
        -------
        list of (int, int)
            Sorted exonic intervals, or empty list when *utr_row* is ``None``.
            When the UTR boundaries extend beyond any annotated exon (e.g. when
            external FASTA-derived UTR annotations are wider than the transcript
            in the GTF), the uncovered portions are treated as exonic and merged
            into the boundary intervals.
        """
        if utr_row is None:
            return []

        utr_low = min(utr_row["start"], utr_row["end"])
        utr_high = max(utr_row["start"], utr_row["end"])

        exons = []
        for exon in self.db.children(gene_feat, featuretype="exon", order_by="start"):
            es = max(exon.start - 1, utr_low)
            ee = min(exon.stop, utr_high)
            if es < ee:
                exons.append((es, ee))

        # If no annotated exon overlaps the UTR region treat the entire UTR as a single exonic interval
        if not exons:
            return [(utr_low, utr_high)]

        # The UTR boundaries (e.g. from external FASTA annotation) may extend
        # beyond the annotated transcript.  The portion of the UTR not covered
        # by any exon feature is implicitly exonic.
        exons = sorted(exons)
        if exons[0][0] > utr_low:
            exons[0] = (utr_low, exons[0][1])
        if exons[-1][1] < utr_high:
            exons[-1] = (exons[-1][0], utr_high)

        return exons

    def _exons_from_utr_introns(
        self,
        gene_feat: gffutils.Feature,
        intron_featuretype: str,
        utr_row: dict | None,
    ) -> list[tuple[int, int]]:
        """Derive exonic intervals within a UTR by subtracting annotated introns.

        Used for GFF3 files (e.g. SGD/yeast) that carry explicit
        ``five_prime_UTR_intron`` / ``three_prime_UTR_intron`` features.

        Parameters
        ----------
        gene_feat:
            Representative gene/transcript feature.
        intron_featuretype:
            GFF feature type for the introns (e.g. ``"five_prime_UTR_intron"``).
        utr_row:
            Row dict from :meth:`_utr5_row` / :meth:`_utr3_row`, or ``None``.

        Returns
        -------
        list of (int, int)
            Sorted exonic intervals covering the UTR with introns removed.
            Returns the whole UTR as a single interval when no introns overlap.
        """
        if utr_row is None:
            return []

        utr_low = min(utr_row["start"], utr_row["end"])
        utr_high = max(utr_row["start"], utr_row["end"])

        introns: list[tuple[int, int]] = []
        for feat in self.db.children(gene_feat, featuretype=intron_featuretype, order_by="start"):
            is_ = max(feat.start - 1, utr_low)
            ie = min(feat.stop, utr_high)
            if is_ < ie:
                introns.append((is_, ie))

        if not introns:
            return [(utr_low, utr_high)]

        introns.sort()
        exons = []
        pos = utr_low
        for is_, ie in introns:
            if pos < is_:
                exons.append((pos, is_))
            pos = max(pos, ie)
        if pos < utr_high:
            exons.append((pos, utr_high))

        return exons
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_to_gene_map(
        self,
        featuretypes: list[str] | None = None,
    ) -> dict[str, str]:
        """Build a mapping from arbitrary feature IDs to gene names.

        Uses the in-memory database to traverse the feature hierarchy and
        returns a dict ``{feature_id: gene_name}`` for every feature of the
        requested type(s) that can be linked to a parent gene.

        The gene name priority follows :func:`_gene_name_from_feature`:
        ``gene_name > gene_id > Name > ID > locus_tag``.

        Parameters
        ----------
        featuretypes:
            List of GFF feature types to include (e.g. ``["transcript",
            "mRNA", "CDS"]``).  Defaults to iterating all features, which
            covers the common case of mapping gffread FASTA transcript IDs to
            gene names.

        Returns
        -------
        dict
            ``{feature_id: gene_name}``
        """
        feat_to_gene: dict[str, str] = {}

        if featuretypes is not None:
            featdb = self.db.features_of_type(featuretypes)
        else:
            featdb = self.db.all_features()

        for feat in featdb:
            feat_id = feat.id

            gene_name = None
            for parent in self.db.parents(feat, featuretype=_GENE_FEATURETYPES):
                gene_name = _gene_name_from_feature(parent)
                if gene_name:
                    break

            if gene_name is None:
                gene_name = _gene_name_from_feature(feat)

            if gene_name is None:
                log.debug(
                    "No gene name found for feature %s (%s); using feature ID.",
                    feat_id, feat.featuretype,
                )
                gene_name = feat_id

            feat_to_gene[feat_id] = gene_name

        return feat_to_gene

    def parse_cds_starts(self) -> dict:
        """Parse CDS start positions from the annotation file.

        Returns a dict keyed by ``(seqid, transcript_id)`` where each value
        is::

            {
                'seqid':     str,   # chromosome / sequence name
                'parent':    str,   # transcript / parent feature ID
                'strand':    str,   # '+' or '-'
                'cds_start': int,   # 0-based genomic start of translation
            }

        For the ``+`` strand *cds_start* is the minimum CDS exon start
        (0-based).  For the ``-`` strand it is the maximum CDS exon end
        (0-based), i.e. the first translated nucleotide when reading 3'→5'.

        Returns
        -------
        dict
            Mapping ``(seqid, transcript_id) -> info_dict`` as described
            above.
        """
        cds_by_parent: dict = {}

        for feat in self.db.features_of_type("CDS"):
            seqid = feat.seqid
            strand = feat.strand
            attrs = feat.attributes

            # Resolve parent/transcript identifier from feature attributes.
            # gffutils stores attribute values as lists; take the first element.
            parent = (
                (attrs.get('Parent') or [''])[0]
                or (attrs.get('parent') or [''])[0]
                or (attrs.get('transcript_id') or [''])[0]
                or (attrs.get('ID') or [''])[0]
                or seqid
            ) or seqid

            # gffutils stores 1-based start and 1-based inclusive end.
            # Convert to 0-based inclusive coordinates to match the original behaviour.
            s0 = feat.start - 1   # 0-based inclusive start
            e0 = feat.end - 1     # 0-based inclusive end (last base)

            key = (seqid, parent)
            if key not in cds_by_parent:
                cds_by_parent[key] = {'strand': strand, 'ranges': [], 'seqid': seqid}
            cds_by_parent[key]['ranges'].append((s0, e0))

        result: dict = {}
        for (seqid, parent), info in cds_by_parent.items():
            strand = info['strand']
            ranges = info['ranges']
            if strand == '+':
                cds_start = min(r[0] for r in ranges)
            else:
                cds_start = max(r[1] for r in ranges)
            result[(seqid, parent)] = {
                'seqid':     seqid,
                'parent':    parent,
                'strand':    strand,
                'cds_start': cds_start,
            }

        return result

    def write_aip_tab(self, outfile: str) -> None:
        """Write the tab file expected by aip.cdsparser.

        Output format (one line per gene, tab-separated):
            gene  chr  strand  cds_length  n_exons  exon1_start  exon1_end  [...]

        Coordinates are 1-based closed intervals, matching the original
        process_gff_sacCer3 / process_gff_ecoli convention.

        Parameters
        ----------
        outfile:
            Path for the output tab file.
        """
        gene_cds = self._cds_features_by_gene()

        if not gene_cds:
            raise ValueError(
                f"No CDS features could be extracted from {self.annotation_file}. "
                "Check that the file contains 'CDS' feature lines."
            )

        with open(outfile, "w") as fh:
            for gene_name, cds_list in gene_cds.items():
                seqid = cds_list[0].seqid
                strand = cds_list[0].strand
                exons = [(f.start, f.end) for f in cds_list]
                cds_length = sum(e - s + 1 for s, e in exons)
                n_exons = len(exons)

                parts = [gene_name, seqid, strand, str(cds_length), str(n_exons)]
                for s, e in exons:
                    parts += [str(s), str(e)]
                fh.write("\t".join(parts) + "\n")

    def extract_cds_sequences(self, genome_fasta: str) -> dict[str, str]:
        """Extract spliced coding sequences for every CDS-containing transcript.

        Pure-Python equivalent of ``gffread -C -w`` followed by trimming each
        transcript to its CDS: the CDS segments of each transcript are fetched
        from *genome_fasta*, concatenated in genomic order and reverse
        complemented on the ``-`` strand.  Stop codons annotated as separate
        features (GTF) are included, see :meth:`_merge_stop_codons_into_cds`.

        Parameters
        ----------
        genome_fasta:
            Path to the genome FASTA file (indexed on the fly if needed).

        Returns
        -------
        dict
            ``{transcript_id: cds_sequence}`` sorted by genomic position
            (sequence, transcript start/end, ID). Transcript IDs can be mapped
            to gene names with :meth:`build_feature_to_gene_map`.
        """
        import pysam
        from .seq_utils import reverse_complement

        tx_cds: dict[str, list[gffutils.Feature]] = {}
        tx_span: dict[str, tuple[int, int]] = {}
        for feat in self.db.features_of_type("CDS"):
            parents = list(self.db.parents(feat, level=1))
            tx = parents[0] if parents else feat
            tx_cds.setdefault(tx.id, []).append(feat)
            tx_span[tx.id] = (tx.start, tx.end)

        # Deterministic genomic order: sequence (first appearance), transcript span, ID.
        seq_rank: dict[str, int] = {}
        for feats in tx_cds.values():
            seq_rank.setdefault(feats[0].seqid, len(seq_rank))
        order = sorted(tx_cds, key=lambda t: (seq_rank[tx_cds[t][0].seqid], *tx_span[t], t))

        cds_seq: dict[str, str] = {}
        with pysam.FastaFile(genome_fasta) as fa:
            for tx_id in order:
                feats = tx_cds[tx_id]
                # Stop codon pieces not absorbed into a CDS (e.g. stop codons split by an intron).
                for stop in self.db.children(tx_id, featuretype="stop_codon"):
                    if not any(f.start <= stop.start and stop.end <= f.end for f in feats):
                        feats.append(stop)
                feats = sorted(feats, key=lambda f: f.start)
                if feats[0].seqid not in fa.references:
                    log.warning("Sequence '%s' of transcript %s not found in %s; skipping.",
                                feats[0].seqid, tx_id, genome_fasta)
                    continue
                seq = "".join(fa.fetch(f.seqid, f.start - 1, f.end) for f in feats)
                if feats[0].strand == "-":
                    seq = reverse_complement(seq)
                    first = feats[-1]
                else:
                    first = feats[0]
                # Like gffread, skip the leading `phase` nt of 5'-incomplete CDSs to stay in frame.
                phase = int(first.frame) if first.frame in ("1", "2") else 0
                cds_seq[tx_id] = seq[phase:].upper()
        return cds_seq

    def parse_utr_exon_intervals(
        self,
        utr5_df: "pd.DataFrame | None" = None,
        utr3_df: "pd.DataFrame | None" = None,
    ) -> dict[str, dict[str, list[tuple[int, int]]]]:
        """Parse exonic intervals within 5'UTR and 3'UTR for every gene.

        The strategy is selected automatically:

        * **Yeast / GFF3 mode** – when ``five_prime_UTR_intron`` features exist,
          exon intervals are derived by subtracting those annotated introns from
          the UTR boundaries.
        * **Human / GTF mode** – otherwise, ``exon`` features from the annotation
          are clipped to the UTR region and used directly as exonic intervals.

        When no UTR is annotated for a gene the corresponding list is empty (the
        caller should treat the whole window as exonic in that case).

        Parameters
        ----------
        utr5_df:
            Optional DataFrame with columns ``gene, chr, start, end, strand,
            length`` containing externally-derived 5'UTR boundaries (e.g. from a
            FASTA annotation such as *SGD_all_ORFs_5prime_UTRs.fsa*).
        utr3_df:
            Same as *utr5_df* but for the 3'UTR.

        Returns
        -------
        dict
            ``{gene_name: {"5UTR": [(start, end), ...], "3UTR": [(start, end), ...]}}``

            All coordinates are 0-based, half-open genomic intervals, sorted
            ascending.
        """
        use_utr_introns = self._has_utr_intron_features()
        if use_utr_introns:
            log.debug(
                "UTR intron features detected (yeast GFF style). "
                "Deriving UTR exon intervals by subtracting five/three_prime_UTR_intron features."
            )
        else:
            log.debug(
                "Deriving UTR exon intervals from 'exon' features (GTF/GFF3 exon mode)."
            )

        # Pre-index the external UTR DataFrames by gene name for O(1) lookup.
        utr5_by_gene: dict = {}
        if utr5_df is not None and not utr5_df.empty:
            for gene_name_key, grp in utr5_df.groupby("gene"):
                utr5_by_gene[gene_name_key] = grp

        utr3_by_gene: dict = {}
        if utr3_df is not None and not utr3_df.empty:
            for gene_name_key, grp in utr3_df.groupby("gene"):
                utr3_by_gene[gene_name_key] = grp

        gene_feats = self._codgene_features_by_gene()
        result: dict[str, dict] = {}

        for gene_name, gene_feat in gene_feats.items():
            seqid = gene_feat.seqid
            strand = gene_feat.strand
            try:
                # 5'UTR row: prefer externally-annotated boundaries when available.
                if gene_name in utr5_by_gene:
                    grp = utr5_by_gene[gene_name]
                    utr5_start = int(grp["start"].min())
                    utr5_end = int(grp["end"].max())
                    utr5_length = utr5_end - utr5_start
                    utr5_row = (
                        dict(gene=gene_name, chr=seqid, start=utr5_start,
                             end=utr5_end, strand=strand, length=utr5_length)
                        if utr5_length > 0 else None
                    )
                else:
                    utr5_row = self._utr5_row(gene_feat, gene_name, seqid, strand)

                # 3'UTR row: prefer externally-annotated boundaries when available.
                if gene_name in utr3_by_gene:
                    grp = utr3_by_gene[gene_name]
                    utr3_start = int(grp["start"].min())
                    utr3_end = int(grp["end"].max())
                    utr3_length = utr3_end - utr3_start
                    utr3_row = (
                        dict(gene=gene_name, chr=seqid, start=utr3_start,
                             end=utr3_end, strand=strand, length=utr3_length)
                        if utr3_length > 0 else None
                    )
                else:
                    utr3_row = self._utr3_row(gene_feat, gene_name, seqid, strand)

            except Exception as exc:
                log.debug("Skipping UTR exon intervals for gene %s: %s", gene_name, exc)
                result[gene_name] = {"5UTR": [], "3UTR": []}
                continue

            if use_utr_introns:
                utr5_exons = self._exons_from_utr_introns(
                    gene_feat, "five_prime_UTR_intron", utr5_row
                )
                utr3_exons = self._exons_from_utr_introns(
                    gene_feat, "three_prime_UTR_intron", utr3_row
                )
            else:
                utr5_exons = self._exons_from_exon_features(gene_feat, utr5_row)
                utr3_exons = self._exons_from_exon_features(gene_feat, utr3_row)

            result[gene_name] = {"5UTR": utr5_exons, "3UTR": utr3_exons}

        log.info(
            "UTR exon intervals parsed for %d genes (%s mode).",
            len(result),
            "utr-intron" if use_utr_introns else "exon",
        )
        return result

    def parse_annotation(self, region: str) -> pd.DataFrame:
        """Return a DataFrame of genomic region boundaries for all genes.

        Parameters
        ----------
        region:
            One of ``"CDS"``, ``"5UTR"``, or ``"3UTR"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``gene, chr, start, end, strand, length``.

            * For ``"CDS"``: ``start``/``end`` are the outermost CDS
              boundaries in 0-based half-open coordinates (BED-style).
            * For ``"5UTR"`` / ``"3UTR"``: ``start``/``end`` span the region
              between the transcript boundary and the start/stop codon.
        """
        if region not in ("CDS", "5UTR", "3UTR"):
            raise ValueError(f"region must be 'CDS', '5UTR', or '3UTR'; got {region!r}")

        log.info("Parsing %s annotations from %s", region, self.annotation_file)
        gene_cds = self._codgene_features_by_gene()

        if not gene_cds:
            raise ValueError(
                f"No CDS features could be extracted from {self.annotation_file}. "
                "Check that the file contains 'CDS' feature lines."
            )

        rows = []

        for gene_name, gene_feat in gene_cds.items():
            seqid = gene_feat.seqid
            strand = gene_feat.strand
            try:
                if region == "CDS":
                    rows.append(self._cds_row(gene_feat, gene_name, seqid, strand))
                elif region == "5UTR":
                    row = self._utr5_row(gene_feat, gene_name, seqid, strand)
                    if row is not None:
                        rows.append(row)
                elif region == "3UTR":
                    row = self._utr3_row(gene_feat, gene_name, seqid, strand)
                    if row is not None:
                        rows.append(row)
            except Exception as exc:
                log.debug("Skipping gene %s (%s): %s", gene_name, region, exc)

        df = pd.DataFrame(rows, columns=["gene", "chr", "start", "end", "strand", "length"])
        return df
