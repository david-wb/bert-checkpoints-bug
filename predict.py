import os

import tensorflow as tf

from model import build_model

dir_path = os.path.dirname(os.path.realpath(__file__))
bert_dir = os.path.join(dir_path, 'uncased_L-12_H-768_A-12')

model = build_model(bert_dir)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('checkpoints')).assert_existing_objects_matched()

input_ids = tf.zeros((1, 384))
input_mask = tf.zeros((1, 384))
segment_ids = tf.zeros((1, 384))

model.predict([input_ids, input_mask, segment_ids])
