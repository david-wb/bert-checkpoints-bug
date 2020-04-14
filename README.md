# bert-checkpoints-bug

# Steps to reproduce

Download the bert pretrained model https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz

Then train to produce checkpoints

```buildoutcfg
python train.py
```

Try to predict. Fails to load the checkpoint.
```buildoutcfg
python predict.py
```
