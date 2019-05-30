# Computing translation scores

## Preprocessing of parallel corpora

See `build_corpus.sh` and `run_sentencepiece.sh` for preprocessing paralell corpora.

## Drop long sentences

Example: Drop sentences with more than 50 words

```shell
zsh drop_long_sentences.sh --in1 parallel_corpora/train/20170418.WN.BA.TC.RE.NO/train.en --in2 parallel_corpora/train/20170418.WN.BA.TC.RE.NO/train.ja --out1 parallel_corpora/train/20170418.WN.BA.TC.RE.NO/train.l50.en --out2 parallel_corpora/train/20170418.WN.BA.TC.RE.NO/train.l50.ja --length 50
```


## Training a seq2seq model using lamtram

[lamtram](https://github.com/neubig/lamtram)

We followed [the tutorial](https://github.com/neubig/nmt-tips) to set hyperparameters.

```shell
CORPUS_DIR=<path to paralell corpora>
MODEL_DIR=<path to output directory>

lamtram/src/lamtram/lamtram-train \
--model_type encatt \
--train_src ${CORPUS_DIR}/train.l50.en.sp8003 \
--train_trg ${CORPUS_DIR}/train.l50.ja.sp8003 \
--dev_src ${CORPUS_DIR}/dev.en.sp8003 \
--dev_trg ${CORPUS_DIR}/dev.ja.sp8003 \
--layers lstm:512:1 \
--trainer adam \
--learning_rate 0.001 \
--minibatch_size 2048 \
--rate_decay 0.5 \
--dropout 0.25 \
--epochs 10 \
--seed 1 \
--model_out ${MODEL_DIR}/encatt.l50.decay.w512.do25.mod \
--eval_every 250000 \
--rate_thresh 0.00025 \
--dynet-seed 1 2>&1 | tee ${MODEL_DIR}/encatt.l50.decay.w512.do25.log && xz ${MODEL_DIR}/encatt.l50.decay.w512.do25.log -v
```

Training log will be written to `${MODEL_DIR}/encatt.l50.decay.w512.do25.log.xz`.
