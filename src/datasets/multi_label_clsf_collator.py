import torch
from torch.nn.utils.rnn import pad_sequence


class MultiLabelClassificationCollator:
    def __init__(self, label2ind, label_sep='|'):
        self.label2ind = label2ind
        self.label_sep = label_sep
    
    def __call__(self, objs):
        names, wavs, labels = list(zip(*objs))
        wavs = pad_sequence(wavs, batch_first=True)

        one_hot_labels = []
        for label_str in labels:
            obj_labels = label_str.split(self.label_sep)
            label_indices = [self.label2ind[label] for label in obj_labels]
            obj_one_hot_labels = torch.zeros(len(self.label2ind))
            obj_one_hot_labels[label_indices] = 1.0
            one_hot_labels.append(obj_one_hot_labels)
        one_hot_labels = pad_sequence(one_hot_labels, batch_first=True)

        return wavs, one_hot_labels
