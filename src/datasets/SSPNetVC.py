import csv
from pathlib import Path
from shutil import unpack_archive

import pandas as pd
import torch
import torchaudio
import wget

URL = 'http://www.dcs.gla.ac.uk/~vincia/datavocalizations/vocalizationcorpus.zip'


class SSPNetVC(torch.utils.data.Dataset):
    def __init__(self, root='data/ssp', csv=None, sr=32_000):
        super().__init__()
        self.root = Path(root)
        if csv is None:
            csv = self.root / 'labels.csv'
        if not self.root.exists() or not csv.exists():
            self.download(self.root)

        self.sr = sr
        self.csv = pd.read_csv(csv)
    
    def __getitem__(self, ind):
        name, labels = self.csv.iloc[ind]
        wav, sr = torchaudio.load(self.root / 'data' / name)
        wav = torchaudio.functional.resample(wav, sr, self.sr).squeeze()
        assert wav.dim() == 1
        return name, wav, labels

    def __len__(self):
        return len(self.csv)

    @staticmethod
    def download(root):
        root = Path(root)
        root.mkdir(exist_ok=True)

        archive = root / 'archive.zip'
        wget.download(URL, out=str(archive))
        unpack_archive(archive, extract_dir=root)
        archive.unlink()

        data = {'Name': [], 'Label': []}
        with open(root / 'labels.txt') as f:
            for line in f:
                if line.startswith('Sample'):
                    # header
                    continue
                fields = line.split(',')

                name = fields[0]

                laughter = ('laughter' in fields)
                filler = ('filler' in fields)
                assert laughter or filler
                if laughter and filler:
                    label = 'Filler|Laughter'
                elif laughter:
                    label = 'Laughter'
                else:
                    label = 'Filler'

                data['Name'].append(f'{name}.wav')
                data['Label'].append(label)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(root / 'labels.csv', index=False, quoting=csv.QUOTE_ALL)
