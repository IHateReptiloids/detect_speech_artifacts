from pathlib import Path

from src.datasets import Event, SSPNetVC


def test_SSPNetVC():
    root = Path('data/ssp')
    if not (root / 'labels.txt').exists():
        SSPNetVC.download(root)
    ds = SSPNetVC(data_path=root / 'data', labels_path=root / 'labels.txt')
    assert ds[6][1] == [Event('filler', ds.LABEL2IND['filler'], 5.457, 5.921),
                        Event('filler', ds.LABEL2IND['filler'], 6.411, 6.566),
                        Event('filler', ds.LABEL2IND['filler'], 7.975, 8.462)]
    assert ds[-1][1] == [
        Event('laughter', ds.LABEL2IND['laughter'], 1.405, 2.093),
        Event('laughter', ds.LABEL2IND['laughter'], 3.625, 4.534)
    ]
    assert ds[1234][0].shape == (352_000,)
