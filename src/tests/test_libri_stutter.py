from pathlib import Path

from src.datasets import Event, LibriStutter


def test_SSPNetVC():
    root = Path('data/libri_stutter')
    if not (root / 'README.txt').exists():
        LibriStutter.download(root)
    ds = LibriStutter(data_path=root)
    assert ds[2][1] == [
        Event(label='phrase repetition', label_idx=4,
              start=0.9, end=2.5332879818594107),
        Event(label='phrase repetition', label_idx=4,
              start=6.533287981859411, end=8.966621315192743),
        Event(label='word repetition', label_idx=3,
              start=14.566621315192743, end=16.15931972789116)
    ]
