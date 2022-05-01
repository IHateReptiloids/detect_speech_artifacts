from src.datasets import ConcatDataset, LibriStutter, SSPNetVC


def test_concat_dataset():
    ls_ds = LibriStutter()
    ssp_ds = SSPNetVC()

    ds = ConcatDataset([ls_ds, ssp_ds])
    assert ds.IND2LABEL == [
        'nothing', 'filler', 'sound repetition', 'word repetition',
        'phrase repetition', 'prolongation', 'laughter'
    ]
    assert ds.LABEL2IND == {
        'nothing': 0, 'filler': 1, 'sound repetition': 2, 'word repetition': 3,
        'phrase repetition': 4, 'prolongation': 5, 'laughter': 6
    }

    ds = ConcatDataset([ssp_ds, ls_ds])
    assert ds.IND2LABEL == [
        'nothing', 'filler', 'laughter', 'sound repetition',
        'word repetition', 'phrase repetition', 'prolongation'
    ]
    assert ds.LABEL2IND == {
        'nothing': 0, 'filler': 1, 'laughter': 2, 'sound repetition': 3,
        'word repetition': 4, 'phrase repetition': 5, 'prolongation': 6
    }
