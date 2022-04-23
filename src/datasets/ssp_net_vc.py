from pathlib import Path
from shutil import unpack_archive

import torch
import torchaudio
import wget

URL = 'http://www.dcs.gla.ac.uk/~vincia/datavocalizations/vocalizationcorpus.zip'


class SSPNetVC(torch.utils.data.Dataset):
    def __init__(self, data_path='data/ssp/data',
                 labels_path='data/ssp/labels.txt', target_sr=32_000):
        super().__init__()
        data_path = Path(data_path)

        self._data = []
        with open(labels_path) as f:
            for line in f:
                if line.startswith('Sample'):
                    # header
                    continue
                fields = line.split(',')

                name = fields[0]
                wav, sr = torchaudio.load(data_path / f'{name}.wav')
                if sr != target_sr:
                    wav = torchaudio.functional.resample(wav, sr, target_sr).squeeze()
                assert wav.dim() == 1

                labels = []
                for i in range(4, len(fields), 3):
                    type_, start, end = fields[i:i + 3]
                    assert type_ in ('filler', 'laughter')
                    labels.append((type_, float(start), float(end)))
                self._data.append((wav, labels))

    def __getitem__(self, ind):
        return self._data[ind]
    
    def __len__(self):
        return len(self._data)
                
    @staticmethod
    def download(root):
        root = Path(root)
        root.mkdir(exist_ok=True, parents=True)

        archive = root / 'archive.zip'
        if archive.exists():
            raise RuntimeError(f'{archive.resolve()} already exists')
        wget.download(URL, out=str(archive))
        unpack_archive(archive, extract_dir=root)
        archive.unlink()
