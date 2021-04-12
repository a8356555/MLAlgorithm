import numpy as np

class Dataset:
    def __init__(self, data, y_col = None):
        self.data = data
        self.features = [col for col in data.dtype.names if col != y_col] if y_col else data.dtype.names[:-1]
        self.iscategory = [self.iscategorical(data[ft]) for ft in self.features]        
        self.labels = data[y_col] if y_col else data[self.features[-1]]
        self.label_name = y_col
        
    def __len__(self):
        return len(self.data)

    def iscategorical(self, col):
        if len(col)/2 > len(np.unique(col)):
            return True
        if isinstance(col[0], str):
            return True
        return False

    def __getitem__(self, idx):
        return self.data[idx]

