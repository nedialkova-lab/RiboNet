"""
This script uses the following models:
- RiNALMo: Penić, R.J., Vlašić, T., Huber, R.G. et al. 
           RiNALMo: general-purpose RNA language models can generalize well on structure prediction tasks. 
           Nat Commun 16, 5671 (2025). https://doi.org/10.1038/s41467-025-60872-5
- SpeciesLM: Karollus, A., Hingerl, J., Gankin, D. et al. 
             Species-aware DNA language models capture regulatory elements and their evolution. 
             Genome Biol 25, 83 (2024). https://doi.org/10.1186/s13059-024-03221-x
"""

import multiprocessing
import logging, os

# model loading can be extended if desired
from transformers import AutoTokenizer, AutoModelForMaskedLM


log = logging.getLogger(__name__)


def tokenize(kmers, tokenizer, maxlen, padsite="right", proxy_species=None):
    '''
    Tokenize k-mer sequences using a language model tokenizer.

    Parameters
    ----------
    kmers : str or list of str
        K-mer sequence(s) to tokenize.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer instance from a language model.
    maxlen : int
        Maximum sequence length for padding/truncation.
    padsite : str, default="right"
        Where to add padding. Options: "right" or "left".
    proxy_species : str, optional
        Species identifier for SpeciesLM models. Ignored for other models.

    Returns
    -------
    list of int
        Token IDs representing the input sequence.
    '''
    if proxy_species is None:
        tokens = tokenizer.encode(kmers, padding="max_length", max_length=maxlen + 2)
        if padsite == "left":
            pad_id = tokenizer.pad_token_id
            padtok = [s for s in tokens if s == pad_id]
            seqtok = [s for s in tokens[1:] if s != pad_id]
            tokens = tokens[:1] + padtok + seqtok
    else:
        if padsite == "left":
            tokens = tokenizer.encode((proxy_species + " " + " ".join(["[PAD]"]*(maxlen-len(kmers))) + " " + " ".join(kmers)).rstrip())
        else:
            tokens = tokenizer.encode((proxy_species + " " +  " ".join(kmers) + " " + " ".join(["[PAD]"]*(maxlen-len(kmers)))).rstrip())
    return tokens


def load_lm(lm, cache_dir=None, model_dir=None, species_proxy=None): 
    """
    Load a model and tokenizer either from a known registry
    or from a custom local path.
    Parameters
    ----------
    lm : str or dict
        The key identifying which model to load, or a dict with keys "UTR5" and "UTR3" specifying the models for both regions
    cache_dir : str, optional
        Has to be the same type as lm if not None. 
        A local directory to use as cache for downloaded models. 
        If None, the default cache directory of the transformers library will be used.
    model_dir : str or dict, optional
        Has to be the same type as lm if not None. 
        A local directory either containing a pretrained model or the path to a directory where the model should 
        additionally be stored after download if locally not available. 
        If None, the model will be downloaded if not cached locally
    species_proxy : str or dict, optional
        A species proxy string to condition the chosen language model on. Ignored for non-SpeciesLM models
    
    Returns
    -------
    list or dict
        If parameter lm is a string: a list containing the model name, tokenizer, model, species proxy, and token size.
        If parameter lm is a dict: a dict with keys "UTR5" and "UTR3", each containing a list as described above.
    """    
    if lm is None:
        return None
    
    elif type(lm) == str: # nucleotide resolution
        if lm.lower() not in model_registry.keys():
            err_message = f"'{lm}' is not a supported model. Available: {list(model_registry.keys())}"
            log.error(err_message)
            raise ValueError(err_message)   

        config = model_registry[lm.lower()]
        func = config["func"]

        return func(name=lm, cache_dir=cache_dir, model_dir=model_dir, species_proxy=species_proxy, **config)
    elif type(lm) == dict: # codon resolution
        lm["UTR5"] = load_lm(lm["UTR5"], 
                            cache_dir=cache_dir,
                            model_dir=model_dir["UTR5"] if model_dir else None, 
                            species_proxy=species_proxy["UTR5"] if species_proxy else None)
        lm["UTR3"] = load_lm(lm["UTR3"], 
                            cache_dir=cache_dir,
                            model_dir=model_dir["UTR3"] if model_dir else None, 
                            species_proxy=species_proxy["UTR3"] if species_proxy else None)

        return lm

    else:
        err_message = "The provided lm object must be a str for nucleotide resolution or a dict containing " \
        "the specified models for 5' and 3' UTR for codon resolution"
        log.error(err_message)
        raise ValueError(err_message)
    


def load_SpeciesLM_model(name, species_proxy, cache_dir=None, model_dir=None, **kwargs):
    '''
    Load a SpeciesLM model and tokenizer.

    Parameters
    ----------
    name : str
        Name identifier for the model.
    species_proxy : str
        Species identifier to condition the model.
    cache_dir : str, optional
        Directory for caching downloaded models.
    model_dir : str, optional
        Local directory containing or to save the pretrained model.
    **kwargs
        Additional arguments including pretrained_name, tok_size, revision.

    Returns
    -------
    list
        [name, tokenizer, model, species_proxy, tok_size]
    '''
    pretrained_name = kwargs["pretrained_name"]
    tok_size = kwargs["tok_size"]
    left_special_tokens = kwargs["left_special_tokens"]
    right_special_tokens = kwargs["right_special_tokens"]
    fixed_length = kwargs["fixed_length"]
    revision = kwargs.get("revision", None)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=revision, cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_dir, revision=revision, cache_dir=cache_dir)

        log.info(f"Loaded {name} model from local {model_dir}")
        return [name.lower(), tokenizer, model, species_proxy, tok_size, left_special_tokens, right_special_tokens, fixed_length]
    except Exception as e:
        if model_dir:
            log.info(f"{name} not found in {model_dir} ({e}). Loading model from cache or remote and saving to {model_dir}")
        else:
            log.info(f"Loading {name} model from cache or remote")

    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name, revision=revision, cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(pretrained_name, revision=revision, cache_dir=cache_dir)
    except Exception as e:
        err_message = (f"Failed to load SpeciesLM model '{name}' from '{pretrained_name}'. "
                       f"Check your internet connection or provide a valid --model_dir path. Error: {e}")
        log.error(err_message)
        raise RuntimeError(err_message) from e

    log.info(f"Loaded {name} model from cache or remote")

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        log.info(f"Saved {name} model to local {model_dir}")

    return [name.lower(), tokenizer, model, species_proxy, tok_size, left_special_tokens, right_special_tokens, fixed_length]

def load_multimolecule_model(name, species_proxy=None, cache_dir=None, model_dir=None, **kwargs):
    """
    Load a model and tokenizer either from a known registry
    or from a custom local path. If model not found locally, download from source.
    
    Parameters
    ----------
    name : str
        The key identifying which model to load.
    model_dir : str, optional
        A local directory containing a pretrained model.
    """ 
    # multimolecule is an optional dependency (only on PyPI), imported on demand
    try:
        import multimolecule
    except ImportError as e:
        err_message = (f"The '{name}' language model requires the optional 'multimolecule' package. "
                       "Install it with `pip install multimolecule` (or `pip install ribonet[rinalmo]`).")
        log.error(err_message)
        raise ImportError(err_message) from e
    model_cls = getattr(multimolecule, kwargs["model_cls"])
    tokenizer_cls = getattr(multimolecule, kwargs["tokenizer_cls"])
    tok_size = kwargs["tok_size"]
    left_special_tokens = kwargs["left_special_tokens"]
    right_special_tokens = kwargs["right_special_tokens"]
    fixed_length = kwargs["fixed_length"]
    pretrained_name = kwargs["pretrained_name"]

    try:
        tokenizer = tokenizer_cls.from_pretrained(model_dir, cache_dir=cache_dir)
        model = model_cls.from_pretrained(model_dir, cache_dir=cache_dir)

        log.info(f"Loaded {name} model from local {model_dir}")
        return [name.lower(), tokenizer, model, species_proxy, tok_size, left_special_tokens, right_special_tokens, fixed_length]
    except Exception as e:
        if model_dir:
            log.info(f"{name} not found in {model_dir} ({e}). Loading model from cache or remote and saving to {model_dir}")
        else:
            log.info(f"Loading {name} model from cache or remote")

    try:
        tokenizer = tokenizer_cls.from_pretrained(pretrained_name, cache_dir=cache_dir)
        model = model_cls.from_pretrained(pretrained_name, cache_dir=cache_dir)
    except Exception as e:
        err_message = (f"Failed to load model '{name}' from '{pretrained_name}'. "
                       f"Check your internet connection or provide a valid --model_dir path. Error: {e}")
        log.error(err_message)
        raise RuntimeError(err_message) from e

    log.info(f"Loaded {name} model from cache or remote")

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        log.info(f"Saved {name} model to local {model_dir}")

    return [name.lower(), tokenizer, model, species_proxy, tok_size, left_special_tokens, right_special_tokens, fixed_length]
# register models here according to the requirements for loading it
model_registry = {
    "rinalmo": {
        "func": load_multimolecule_model,
        "tokenizer_cls": "RnaTokenizer",
        "model_cls": "RiNALMoModel",
        "pretrained_name": "multimolecule/rinalmo-giga",
        "tok_size": 1024,
        "left_special_tokens": 1,
        "right_special_tokens": 1,
        "fixed_length": False,  # trained with variable-length padded inputs
        },
    "specieslm5": {
        "func": load_SpeciesLM_model,
        "pretrained_name": "gagneurlab/SpeciesLM",
        "revision": "1052abf26a702a2930311d2ed19157266d321e2d",  # pinned commit of the "upstream_species_lm" branch
        "tok_size": 1001,
        "left_special_tokens": 2, # includes species token
        "right_special_tokens": 1,
        "fixed_length": True,   # trained exclusively on 1001-token fixed-length sequences
    },
    "specieslm3": {
        "func": load_SpeciesLM_model,
        "pretrained_name": "gagneurlab/SpeciesLM",
        "revision": "742e7928895f71b30fada028d51bf63537a61cf8",  # pinned commit of the "downstream_species_lm" branch
        "tok_size": 301,
        "left_special_tokens": 2, # includes species token
        "right_special_tokens": 1,
        "fixed_length": True,   # trained exclusively on 301-token fixed-length sequences
    }
}

def extract_lm_name(lm):
    '''
    Extract the name(s) of the language model(s) from the loaded model object.

    Parameters
    ----------
    lm : list, dict, or None
        Loaded language model object.

    Returns
    -------
    str, dict, or None
        Model name(s) or None if no model loaded.

    Raises
    ------
    ValueError
        If lm has an invalid type.
    '''
    if type(lm) == list:
        return lm[0]
    elif type(lm) == dict:
        return {"UTR5": extract_lm_name(lm["UTR5"]), "UTR3": extract_lm_name(lm["UTR3"])}
    elif lm is None:
        return None
    else:
        raise ValueError("The provided lm object must be a list for nucleotide resolution or a dict containing " \
            "the specified models as two lists for 5' and 3' UTR for codon resolution")


def number_speciesLM(lm):
    '''
    Count how many SpeciesLM models are being used.

    Parameters
    ----------
    lm : str, dict, or None
        Language model specification.

    Returns
    -------
    int or bool
        Number of SpeciesLM models (0, 1, or 2), or False if invalid input.
    '''
    # extract registered speciesLM models
    models = [key for key in model_registry.keys() if 'specieslm' in key.lower()]
    # return number of speciesLM models used
    if type(lm) == str:
        return int('specieslm' in lm.lower())
    elif type(lm) == dict:
        return int(lm["UTR5"].lower() in models) + int(lm["UTR3"].lower() in models)
    else:
        return False


def autoselect_workers():
    '''
    Automatically select an appropriate number of worker processes for data loading.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Number of worker processes, minimum of (CPU count - 1) and 8.
    '''
    try:
        # On Linux/HPC, sched_getaffinity returns the CPUs actually allocated
        # by the job scheduler (e.g. SLURM), which may be far fewer than the
        # machine total returned by cpu_count().
        import os
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = multiprocessing.cpu_count()
    return max(1, min(available - 1, 8))
