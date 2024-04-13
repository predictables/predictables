import torch


class NAICSDataLoader(torch.utils.data.DataLoader):
    """A data loader for the NAICS embedding model.

    Each iteration of the dataloader should return a tuple of the form:
    (train_X, train_y, val_X, val_y)
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        super(NAICSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self.dataset[i]

    def __len__(self):
        return len(self.dataset)