from pathlib import Path
import shutil

from patoolib import extract_archive
import requests
import torch
import torchaudio

from .event import Event

URLS = [
    'https://dataverse.scholarsportal.info/api/access/datafile/270789',
    'https://dataverse.scholarsportal.info/api/access/datafile/270790',
    'https://dataverse.scholarsportal.info/api/access/datafile/270899',
    'https://dataverse.scholarsportal.info/api/access/datafile/270788'
]


class LibriStutter(torch.utils.data.Dataset):
    NUM_CLASSES = 6

    IND2LABEL = ['nothing', 'filler', 'sound repetition',
                 'word repetition', 'phrase repetition', 'prolongation']
    LABEL2IND = {
        'nothing': 0, 'filler': 1, 'sound repetition': 2,
        'word repetition': 3, 'phrase repetition': 4, 'prolongation': 5
    }

    def __init__(self, data_path='data/libri_stutter/',
                 labels_path=None, target_sr=32_000):
        super().__init__()
        data_path = Path(data_path)
        self.annotations_dir = data_path / 'LibriStutter Annotations'
        self.audio_dir = data_path / 'LibriStutter Audio'
        self.target_sr = target_sr

        if labels_path is not None:
            with open(labels_path, 'r') as f:
                self.rel_paths = list(map(lambda s: s.rstrip('\n'),
                                          f.readlines()))
        else:
            self.rel_paths = [p.relative_to(self.audio_dir).with_suffix('')
                              for p in self.audio_dir.glob('*/*/*')]
            self.rel_paths.sort()
    
    def __getitem__(self, ind):
        p = self.rel_paths[ind]
        wav, sr = torchaudio.load((self.audio_dir / p).with_suffix('.flac'))
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.squeeze().numpy()
        assert wav.ndim == 1

        events = []
        with ((self.annotations_dir / p).with_suffix('.csv')).open('r') as f:
            for line in f:
                if line.startswith('STUTTER'):
                    _, start, end, type_ = line.split(',')
                    start, end, type_ = float(start), float(end), int(type_)
                    events.append(Event(self.IND2LABEL[type_], type_,
                                        start, end))
        return wav, events
    
    def __len__(self):
        return len(self.rel_paths)

    @staticmethod
    def download(root):
        root = Path(root)
        annotations_dir = root / 'LibriStutter Annotations'
        annotations_dir.mkdir(exist_ok=True, parents=True)
        audio_dir = root / 'LibriStutter Audio'
        audio_dir.mkdir(exist_ok=True, parents=True)

        for part in range(3):
            archive = root / 'archive.rar'
            if archive.exists():
                raise RuntimeError(f'{archive.resolve()} already exists')
            with archive.open('wb') as f:
                f.write(requests.get(URLS[part]).content)

            extract_archive(archive, outdir=root)
            archive.unlink()

            dir_ = root / f'LibriStutter Part {part + 1}'
            for child in (dir_ / 'LibriStutter Annotations').iterdir():
                child.rename(annotations_dir / child.name)
            for child in (dir_ / 'LibriStutter Audio').iterdir():
                child.rename(audio_dir / child.name)
            shutil.rmtree(dir_)

        readme = root / 'README.txt'
        with readme.open('wb') as f:
            f.write(requests.get(URLS[-1]).content)
