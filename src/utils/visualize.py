import io

import matplotlib.pyplot as plt
import numpy as np


def visualize(wav, pred, y_aligned, ind2label):
    if not wav.dim() == pred.dim() == y_aligned.dim() == 1:
        raise ValueError('All arguments must be one-dimensional')
    if len(pred) != len(y_aligned):
        raise ValueError('pred and y_aligned must have same length')

    colors = ['blue', 'green', 'red', 'cyan',
              'magenta', 'yellow', 'black']
    fig = plt.figure(figsize=(20, 5), dpi=100)
    for i in range(len(pred)):
        wav_part = wav[len(wav) // len(pred) * i:len(wav) // len(pred) * (i + 1)]
        if pred[i] != y_aligned[i]:
            plt.scatter(
                len(wav) // len(pred) * (i + 0.5) / 16_000,
                0,
                c=colors[pred[i].item()],
                zorder=1,
                s=10
            )
        plt.plot(
            np.arange(len(wav) // len(pred) * i, len(wav) // len(pred) * (i + 1)) / 16_000,
            wav_part,
            colors[y_aligned[i].item()],
            zorder=0
        )
    for i in range(len(ind2label)):
        plt.scatter(1, 0, s=10, c=colors[i], label=ind2label[i], zorder=-1)
    plt.legend()
    plt.xlabel('Seconds')
    plt.ylabel('Amplitude')

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=100)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr
