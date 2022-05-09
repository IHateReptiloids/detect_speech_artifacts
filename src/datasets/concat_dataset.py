from collections import OrderedDict

import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.LABEL2IND = OrderedDict()
        for ds in datasets:
            for label in ds.IND2LABEL:
                if label not in self.LABEL2IND:
                    self.LABEL2IND[label] = len(self.LABEL2IND)
        self.IND2LABEL = list(self.LABEL2IND.keys())
        self.NUM_CLASSES = len(self.IND2LABEL)

        self.ds = torch.utils.data.ConcatDataset(datasets)
    
    def __getitem__(self, ind):
        wav, events = self.ds[ind]
        for event in events:
            event.label_idx = self.LABEL2IND[event.label]
        return wav, events
    
    def __len__(self):
        return len(self.ds)
