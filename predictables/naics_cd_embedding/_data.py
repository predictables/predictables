from sklearn.preprocessing import OneHotEncoder
import torch
import pandas as pd

class NAICSDataset(torch.utils.data.Dataset):
    """Custom Dataset for handling NAICS data with one-hot encoding of categorical NAICS codes."""
    def __init__(self, dataframe):
        self.encoder = OneHotEncoder(sparse=False)
        # Assuming 'naics_2_cd' through 'naics_6_cd' need encoding
        naics_features = dataframe.filter(regex='^naics_')
        encoded_features = self.encoder.fit_transform(naics_features)
        
        # Convert encoded features to tensor
        self.features = torch.tensor(encoded_features, dtype=torch.float32)
        
        # Assuming 'target' as labels
        self.labels = torch.tensor(dataframe['target'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NAICSDataLoader(torch.utils.data.DataLoader):
    """A data loader for the NAICS embedding model."""
    def __init__(self, dataframe: pd.DataFrame, batch_size: int = 32, shuffle: bool = True):
        # Convert the DataFrame into a Dataset
        dataset = NAICSDataset(dataframe)
        super(NAICSDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)


