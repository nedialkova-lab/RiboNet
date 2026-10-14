import torch 
import numpy as np

def multinomial_nll(profile_pred, # logits: B, n_tracks, L_seq
                    profile_true, # actual counts: B, n_tracks, L_seq, reps
                    eps=1/1000,
                    clamp_true=None):
    '''
    Compute multinomial negative log-likelihood loss for profile predictions.

    Parameters
    ----------
    profile_pred : torch.Tensor
        Predicted profile logits, shape (batch, n_tracks, seq_len).
    profile_true : torch.Tensor
        True profile counts, shape (batch, n_tracks, seq_len, n_replicates).
    eps : float, default=1/1000
        Small epsilon to add to true counts for numerical stability.
    clamp_true : float, optional
        Maximum value to clamp true counts. If None, no clamping is applied.

    Returns
    -------
    torch.Tensor
        Scalar loss value averaged over all valid replicates.
    '''
    profile_true = profile_true + eps
    if clamp_true:
        profile_true = torch.clamp(profile_true, min=0, max=clamp_true)
    
    # compute the scaling constant
    log_fact_sum = torch.lgamma(torch.sum(profile_true, dim=2) + 1)
    log_prod_fact = torch.sum(torch.lgamma(profile_true + 1), dim=2)
    
    # compute the profile loss term
    # convert logits into log probabilities
    logps = torch.nn.functional.log_softmax(profile_pred, dim=2)
    
    loss = 0
    nrep = 0
    for n in range(profile_true.shape[3]):
        nanmask = torch.isnan(log_fact_sum[:,:,n]).sum(axis=0)==0
        # multiply log probabilities with the true profile
        log_prod_exp = torch.sum(profile_true[:,nanmask,:,n] * logps[:,nanmask,:], dim=2)
        loss += (-log_fact_sum[:,nanmask,n] + log_prod_fact[:,nanmask,n] - log_prod_exp).sum()
        nrep += nanmask.sum()
    return loss/nrep

def profile_ce_loss(profile_pred, # logits: B, n_tracks, L_seq
                    profile_true, # actual counts: B, n_tracks, L_seq, reps
                    eps=1/1000):
    '''
    Compute cross-entropy loss for profile predictions.

    Parameters
    ----------
    profile_pred : torch.Tensor
        Predicted profile logits, shape (batch, n_tracks, seq_len).
    profile_true : torch.Tensor
        True profile counts, shape (batch, n_tracks, seq_len, n_replicates).
    eps : float, default=1/1000
        Small epsilon to add to true counts for numerical stability.

    Returns
    -------
    torch.Tensor
        Scalar loss value averaged over all valid replicates.
    '''
    profile_true = profile_true + eps
    profile_true = profile_true/profile_true.nansum(axis=2,keepdim=True)
    
    shape_tuple = (profile_true.shape[0]*profile_true.shape[1], profile_true.shape[2])
    reshaped_pred = profile_pred.reshape(shape_tuple)

    loss = 0
    nrep = 0
    for n in range(profile_true.shape[3]):
        reshaped_true = profile_true[:,:,:,n].reshape(shape_tuple)
        nanmask = (~torch.isnan(reshaped_true)).sum(axis=-1)>0
        reshaped_pred_temp = reshaped_pred[nanmask,:]
        reshaped_true = reshaped_true[nanmask,:]
        loss += torch.nn.functional.cross_entropy(torch.where(reshaped_true.isnan(), 0.0, reshaped_pred_temp),
                                                  torch.where(reshaped_true.isnan(), 0.0, reshaped_true), reduction="sum")
        nrep += nanmask.sum()
    return loss/nrep

def count_mse_loss(count_pred, # log prediction: B, n_tracks
                   profile_true, # actual counts: B, n_tracks, L_seq, reps
                   tot_reads,
                   utr_size):
    '''
    Compute mean squared error loss for total count predictions.

    Parameters
    ----------
    count_pred : torch.Tensor
        Predicted log counts, shape (batch, n_tracks).
    profile_true : torch.Tensor
        True profile counts, shape (batch, n_tracks, seq_len, n_replicates).
    tot_reads : torch.Tensor
        Total reads per library for normalization.
    utr_size : list of int or torch.Tensor
        If list: [utr5_poolsize, utr3_poolsize] for codon resolution.
        If Tensor: annotation array for nucleotide resolution.

    Returns
    -------
    torch.Tensor
        Scalar MSE loss averaged over all valid replicates.
    '''
    if type(utr_size) == list: # codon resolution
        # use the given utr length (which is the same for all seqs)
        # and set the utr regions to nan
        profile_true = profile_true[:,:,utr_size[0]:-utr_size[1],:]
    elif type(utr_size) == torch.Tensor: # nucleotide resolution
        # use the x["annotation"] from the dataset to set all utr regions to nan
        utr_mask = utr_size >= 4
        utr_mask = utr_mask.unsqueeze(1).unsqueeze(-1).expand_as(profile_true)
        profile_true[utr_mask] = torch.nan
    else:
        raise ValueError("utr_size has to be either a list of two ints or" \
        " a annotation of the current batch of genes where utr nucleotides " \
        "are annotated as ints >= 4")
    rpms = (profile_true.nansum(axis=2))*1e6/tot_reads # B, n_tracks, reps
    rpms[rpms<1] = np.nan
    log_true = torch.log(rpms)
    flatpred = torch.flatten(count_pred)
    rnanan = (~torch.isnan(flatpred)) # this will only happen if RNAseq TPM <1
    loss = 0
    nrep = 0
    for n in range(profile_true.shape[3]):
        flattrue = torch.flatten(log_true[:,:,n])
        nanmask = torch.logical_and(~torch.isnan(flattrue), rnanan)
        loss += torch.nn.functional.mse_loss(flatpred[nanmask], flattrue[nanmask], reduction="sum")
        nrep += nanmask.sum()

    return loss/nrep

def total_bpnet_loss(preds,
                     labels,
                     tot_reads,
                     alpha, 
                     beta,
                     utr_size,
                     ablate_utr5=False,
                     ablate_utr3=False,
                     profile_loss_type="ce",
                     eps=1/1000,
                     clamp_true=None):
    '''
    Compute total BPNet-style hybrid loss combining profile and count losses.

    Parameters
    ----------
    preds : tuple of torch.Tensor
        Tuple of (profile_pred, count_pred) from model output.
    labels : torch.Tensor
        True profile counts, shape (batch, n_tracks, seq_len, n_replicates).
    tot_reads : torch.Tensor
        Total reads per library for normalization.
    alpha : float
        Weight for count loss component.
    beta : float
        Weight for profile loss component.
    utr_size : list of int or torch.Tensor
        UTR size specification for masking.
    ablate_utr5 : bool, default=False
        If True, restrict the profile loss to non-5'UTR positions so that the
        model is not supervised on UTR profile output when UTR input is zeroed.
    ablate_utr3 : bool, default=False
        If True, restrict the profile loss to non-3'UTR positions.
    profile_loss_type : str, default="ce"
        Type of profile loss. Options: "ce" (cross-entropy) or "multinomial".
    eps : float, default=1/1000
        Small epsilon for numerical stability.
    clamp_true : float, optional
        Maximum value to clamp true counts (only for multinomial loss).

    Returns
    -------
    torch.Tensor
        Scalar total loss: beta * profile_loss + alpha * count_loss.
    '''
    profile_pred, count_pred = preds
    profile_true = labels
    
    count_loss = count_mse_loss(count_pred, profile_true, tot_reads, utr_size)

    # When UTR input is ablated, also restrict the profile loss to the non-ablated
    # region so the model is not trained to predict UTR profiles without UTR sequence info.
    if (ablate_utr5 or ablate_utr3) and type(utr_size) == list:
        u5 = utr_size[0] if ablate_utr5 else 0
        u3 = utr_size[1] if ablate_utr3 else 0
        p_pred = profile_pred[:, :, u5 : (profile_pred.size(2) - u3 if u3 > 0 else None)]
        p_true = profile_true[:, :, u5 : (profile_true.size(2) - u3 if u3 > 0 else None), :]
    else:
        p_pred = profile_pred
        p_true = profile_true

    if profile_loss_type == "multinomial":
        profile_loss = multinomial_nll(p_pred, p_true, eps=eps, clamp_true=clamp_true)
    elif profile_loss_type == "ce":
        profile_loss = profile_ce_loss(p_pred, p_true, eps=eps)
    else:
        assert False, "invalid loss specified"
    
    #wandb.log({"profile_loss": profile_loss, "count_loss": count_loss})

    loss = (beta * profile_loss) + (alpha * count_loss)
    return loss