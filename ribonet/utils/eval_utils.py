try:
    import wandb
except ImportError:  # optional, only needed with use_wandb=True
    wandb = None
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

def eval_count_pred(xc, testdf, genelength, use_wandb):
    count_correlation = {}
    if use_wandb:
        table = wandb.Table(columns=[
            "condition", "pearson_TRUE", "pearson_PRED", "pearson_DIFF"
        ])

    def _pearson(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 2:
            return np.nan
        return np.corrcoef(a[mask], b[mask])[0, 1]

    for n, cond in enumerate(testdf.columns.unique("cond")):
        tempdf = pd.melt(
            pd.concat(
                [testdf[cond]["count_pred"],
                pd.DataFrame(testdf[cond]["rpm_true"].to_list(), index=testdf.index)],
                axis=1
            ),
            id_vars="count_pred",
            value_name="rpm_true",
            ignore_index=False
        ).dropna()

        trues = tempdf["rpm_true"].values
        # Length-normalised log RNA: log(RNA+1) - log(CDS_len), one value per replicate row
        rnas = (np.log(np.stack(xc.loc[tempdf.index, "rna"])[:, n] + 1)
                + np.log(genelength.loc[tempdf.index].values))

        r_true = _pearson(trues, rnas)
        r_pred = _pearson(tempdf["rpm_true"].values, tempdf["count_pred"].values)

        count_correlation[cond] = {}
        count_correlation[cond]["prediction"] = r_pred
        count_correlation[cond]["rna_abundance"] = r_true

        if use_wandb:
            table.add_data(cond, r_true, r_pred, r_pred - r_true)
    if use_wandb:
        wandb.log({"count_corr_table": table})
    return count_correlation

def eval_explvar(xc, testdf, train_xc, traindf, genelength, use_wandb):
    all_trues, all_preds, all_rnas = [], [], []

    for n, cond in enumerate(testdf.columns.unique("cond")):
        tempdf = pd.melt(
            pd.concat(
                [testdf[cond]["count_pred"],
                 pd.DataFrame(testdf[cond]["rpm_true"].to_list(), index=testdf.index)],
                axis=1
            ),
            id_vars="count_pred",
            value_name="rpm_true",
            ignore_index=False
        ).dropna()

        all_trues.append(tempdf["rpm_true"].values)
        all_preds.append(tempdf["count_pred"].values)

        # Fit linear model for RNA baseline
        train_rpm_series = pd.DataFrame(traindf[cond]["rpm_true"].to_list(), index=traindf.index).stack().rename("rpm_true").reset_index(level=1, drop=True).dropna()
        train_rna = np.log(np.stack(train_xc.loc[train_rpm_series.index, "rna"])[:, n] + 1)
        train_rpm_norm = (train_rpm_series.values - np.log(genelength.loc[train_rpm_series.index].values))
        isnan = np.isnan(train_rpm_norm) | np.isnan(train_rna)
        reg = LinearRegression().fit(train_rna[~isnan].reshape(-1, 1), train_rpm_norm[~isnan])
        test_rna = np.log(np.stack(xc.loc[tempdf.index, "rna"])[:, n] + 1)
        rna_pred = (reg.predict(test_rna.reshape(-1, 1))+ np.log(genelength.loc[tempdf.index].values))
        all_rnas.append(rna_pred)

    trues = np.concatenate(all_trues)
    preds = np.concatenate(all_preds)
    rnas  = np.concatenate(all_rnas)

    meanval = np.nanmean(trues)
    totvar  = np.nansum((trues - meanval) ** 2)

    explvar     = (totvar - np.nansum((trues - preds) ** 2)) * 100 / totvar
    rna_explvar = (totvar - np.nansum((trues - rnas)  ** 2)) * 100 / totvar

    if use_wandb:
        wandb.log({"explained_variance_ribonet": explvar,
                   "explained_variance_rna_baseline": rna_explvar})

    return {"explained_variance_ribonet": explvar, "explained_variance_rna_baseline": rna_explvar}

def eval_prof_corr(counts_per_nt, conditions, use_wandb):
    profiling_correlation = {}

    if use_wandb: 
        table_corr = wandb.Table(columns=["condition", "prof_corr", "replicate_corr"])
        table_mse = wandb.Table(columns=["condition", "prof_mse"])

    for cond in conditions:
        trues = np.log2(counts_per_nt.loc[counts_per_nt.cond==cond,"counts_true"]*10000 + 32)-5
        preds = np.log2(counts_per_nt.loc[counts_per_nt.cond==cond,"counts_pred"]*10000 + 32)-5
        norm_cor = stats.spearmanr(trues, preds, nan_policy="omit")[0]
        mse = np.nanmean((trues - preds)**2)

        conddf = counts_per_nt.loc[counts_per_nt.cond == cond] \
                .pivot(index=["gene", "pos"], columns="rep", values="counts_true")

        # Skip if fewer than 2 replicates
        if conddf.shape[1] < 2:
            rep_corr = np.nan
        else:
            conddf = np.log2(conddf * 10000 + 32) - 5
            corrmat = conddf.corr(method="spearman")
            # Take mean of off-diagonal (exclude self-correlation)
            rep_corr = np.nanmean(corrmat.values[np.triu_indices_from(corrmat, k=1)])
        
        profiling_correlation[cond] = {}
        profiling_correlation[cond]["prediction_correlation"] = norm_cor
        profiling_correlation[cond]["replicate_correlation"] = rep_corr

        if use_wandb: 
            table_corr.add_data(cond, norm_cor, rep_corr)
            table_mse.add_data(cond, mse)
    
    if use_wandb: 
        wandb.log({
            "profile_corr": table_corr,
            "profile_mse": table_mse
        })
    return profiling_correlation