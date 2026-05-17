import torch
from torch.utils.data import Dataset, DataLoader, random_split



# Define the dataset class
class DFPDataSet(Dataset):
    def __init__(self, df, col_config, embd_mod_name, device):
        super().__init__()
        ...

    def __len__(self):
        ...
    
    def __getitem__(self, index):
        ...

# Data Loaders
def build_dataloaders(
        dataset: DFPDataSet,
        val_fraction: float=.15,
        batch_size: int=256,
        num_workers: int=8,
        seed: int=42
):
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0)) # refer: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))

    return train_loader, val_loader


# Training utils
