import torch


class NAICSDataLoader(torch.utils.data.DataLoader):
    """A data loader for the NAICS embedding model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)