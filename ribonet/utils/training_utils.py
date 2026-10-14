import torch


def train_epoch(model, train_dataloader, optimizer, loss_fn, tot_reads, utr_pool_size, device):
    """
    Train the model for a single epoch over the provided training dataloader.

    Parameters
    ----------
    model : RiboNet
            Either a nucleotide or codon resolution RiboNet
    train_dataloader : torch.utils.data.DataLoader
            DataLoader producing training batches, containing either a
            RiboDataset_nt or a RiboDataset_cod
    optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters. .
    loss_fn : callable
            Loss function, e.g. BPNet loss
    tot_reads : torch.Tensor
            Total RNA counts
    utr_pool_size : Optional[List] or None
            - if codon resolution: a list containing the pooled 5'- and 3'-UTR size
            as ints
            - if nucleotide resolution:
            None. The UTR lengths of each gene are extracted from the dataloaders dataset
    device : torch.device or str
            Device to which inputs, targets and the model will be moved

    Returns
    -------
    float
            The average training loss per example over the entire epoch. 
    """
    device = device.type if isinstance(device, torch.device) else device
    model.train()
    model.to(device)
    train_loss = 0.0

    for x, y in train_dataloader:
        for k,v in x.items():
            x[k] = v.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device):
            outputs = model(x)

            # Compute loss between outputs and y.
            # For both resolutions the MSE loss is only calculated for coding sequence 
            # - For codon resolution the UTRs are the same for all seqs and given by utr_pool_size
            # - For Nucleotide they vary for each sequence -> use annotation from dataset
            loss = loss_fn(outputs, y, tot_reads, utr_size=utr_pool_size if utr_pool_size else x["annotation"])
        
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * x["rna"].size(0)

    train_loss = train_loss / len(train_dataloader.dataset) 

    return train_loss


def evaluate(model, val_dataloader, loss_fn, tot_reads, utr_pool_size, device):
    '''
    Evaluate the model on a validation dataset.

    Parameters
    ----------
    model : RiboNet
        Either a nucleotide or codon resolution RiboNet model.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader producing validation batches.
    loss_fn : callable
        Loss function (e.g., BPNet loss).
    tot_reads : torch.Tensor
        Total RNA counts per library.
    utr_pool_size : list of int or None
        If codon resolution: [utr5_poolsize, utr3_poolsize].
        If nucleotide resolution: None.
    device : torch.device or str
        Device for computation.

    Returns
    -------
    float
        Average validation loss per example.
    '''
    device = device.type if isinstance(device, torch.device) else device
    model.eval()
    model.to(device)
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_dataloader:
            for k,v in x.items():
                x[k] = v.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.autocast(device):
                outputs = model(x)
                loss = loss_fn(outputs, y, tot_reads, utr_size=utr_pool_size if utr_pool_size else x["annotation"])

            val_loss += loss.item() * x["rna"].size(0)

    val_loss = val_loss / len(val_dataloader.dataset)

    return val_loss


def make_prediction(model, test_dataloader, device):
    '''
    Generate predictions using a trained model on a test dataset.

    Parameters
    ----------
    model : RiboNet
        Trained RiboNet model (nucleotide or codon resolution).
    test_dataloader : torch.utils.data.DataLoader
        DataLoader producing test batches.
    device : torch.device or str
        Device for computation.

    Returns
    -------
    tuple of torch.Tensor
        (profile_predictions, count_predictions) where:
        - profile_predictions: softmax probabilities, shape (n_samples, n_tracks, seq_len)
        - count_predictions: log counts, shape (n_samples, n_tracks)
    '''
    device = device.type if isinstance(device, torch.device) else device
    profile_pred_list = []
    count_pred_list = []

    model.eval()
    model.to(device)

    with torch.no_grad():
      for x_test, y_test in test_dataloader:
            for k,v in x_test.items():
                x_test[k] = v.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)

            with torch.autocast(device):
                profile_pred, count_pred = model(x_test)

            profile_pred_list.append(torch.nn.functional.softmax(profile_pred, dim=2))
            count_pred_list.append(count_pred)

    profile_pred_tensor = torch.cat(profile_pred_list, dim=0)
    count_pred_tensor = torch.cat(count_pred_list, dim=0)

    return profile_pred_tensor, count_pred_tensor
