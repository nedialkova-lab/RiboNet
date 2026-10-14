import torch
from torch.utils.data import Dataset

class RiboDataset_nt(Dataset):
    def __init__(self, seq, rna, annotation, y):
        self.seq = torch.tensor(seq, dtype = torch.int64)
        self.rna = torch.tensor(rna, dtype = torch.float64)
        self.annotation = torch.tensor(annotation, dtype = torch.int64)
        self.y = torch.tensor(y, dtype = torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = {"seq":self.seq[index],
             "rna":self.rna[index],
             "annotation":self.annotation[index]}
        y = self.y[index]
        return x, y
    
class RiboDataset_cod(Dataset):
    def __init__(self, cds, utr5, utr3, rna, pos, y):
        self.cds = torch.tensor(cds, dtype = torch.int64)
        self.utr5 = torch.tensor(utr5, dtype = torch.int64)
        self.utr3 = torch.tensor(utr3, dtype = torch.int64)
        self.rna = torch.tensor(rna, dtype = torch.float32)
        self.pos = torch.tensor(pos, dtype = torch.int64)
        self.y = torch.tensor(y, dtype = torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = {"cds":self.cds[index],
             "utr5":self.utr5[index],
             "utr3":self.utr3[index],
             "rna":self.rna[index],
             "pos":self.pos[index]}
        y = self.y[index]
        return x, y