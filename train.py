import os

import tensorflow as tf

from model import build_model

dir_path = os.path.dirname(os.path.realpath(__file__))
bert_dir = os.path.join(dir_path, 'uncased_L-12_H-768_A-12')

model = build_model(bert_dir)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(os.path.join(bert_dir, 'bert_model.ckpt')).expect_partial()

optimizer = tf.optimizers.Adam()
model.compile(optimizer=optimizer, loss=[tf.losses.BinaryCrossentropy()])

input_ids = tf.zeros((1, 384))
input_mask = tf.zeros((1, 384))
segment_ids = tf.zeros((1, 384))

y = tf.zeros((1, 1))


checkpoint_path = "checkpoints/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(x=[input_ids, input_mask, segment_ids], y=y, epochs=3, callbacks=[cp_callback])
