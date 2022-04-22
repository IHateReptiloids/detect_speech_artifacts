from pathlib import Path

from src.datasets import SSPNetVC


def test_SSPNetVC():
    root = Path('data/ssp')
    if not (root / 'labels.txt').exists():
        SSPNetVC.download(root)
    ds = SSPNetVC(lambda x, y: y, data_path=root / 'data',
                  labels_path=root / 'labels.txt')
    assert ds[6][1] == [('filler', 5.457, 5.921),
                        ('filler', 6.411, 6.566),
                        ('filler', 7.975, 8.462)]
    assert ds[-1][1] == [('laughter', 1.405, 2.093),
                         ('laughter', 3.625, 4.534)]
    assert ds[1234][0].shape == (352_000,)
