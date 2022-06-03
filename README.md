# Deep Learning for Prosodic Features Detection in Human Speech

[Wandb project](https://wandb.ai/_username_/diploma)

## SSPNet Vocalization Corpus

| Model                          | Laughter F-score     | Filler F-score     |
| ------------------------------ | -------------------- | ------------------ |
| wav2vec 2.0 + BLSTM            | 0.5454               | **0.6921**         |
| BCResNet                       | 0.4871               | 0.667              |
| BCResNet + BLSTM               | 0.5019               | 0.6733             |
| wav2vec 2.0 + BCResNet + BLSTM | **0.5515**           | 0.6658             |

## LibriStutter
| Model                          | Prolongation F-score     | Phrase repetition F-score     | Sound repetition F-score     | Word repetition F-score     |
| ------------------------------ | ------------------------ | ----------------------------- | ---------------------------- | --------------------------- |
| wav2vec 2.0 + BLSTM            | 0.6783                   | 0.5598                        | 0.616                        | **0.5025**                  |
| BCResNet + BLSTM               | **0.8078**               | 0.6013                        | **0.7275**                   | 0.4799                      |
| wav2vec 2.0 + BCResNet + BLSTM | 0.6167                   | **0.6635**                    | 0.7124                       | 0.442                       |
