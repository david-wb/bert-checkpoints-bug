import os

import tensorflow as tf
from official.nlp.bert.bert_models import get_transformer_encoder
from official.nlp.bert.configs import BertConfig
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


def build_model(bert_dir):
    max_seq_len = 384

    bert_config = BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    bert_encoder = get_transformer_encoder(bert_config, max_seq_len)

    input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input_mask')
    segment_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='segment_ids')

    bert_inputs = [input_ids, input_mask, segment_ids]
    bert_sequence_output, bert_pooled_output = bert_encoder(bert_inputs)

    out = Dense(1, activation='sigmoid', name='out')(bert_pooled_output)
    return Model(inputs=bert_inputs, outputs=[out])


dir_path = os.path.dirname(os.path.realpath(__file__))
bert_dir = os.path.join(dir_path, 'uncased_L-12_H-768_A-12')

# TRAIN

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

# PREDICT

model = build_model(bert_dir)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('checkpoints')).assert_existing_objects_matched()  # <-- ERROR

input_ids = tf.zeros((1, 384))
input_mask = tf.zeros((1, 384))
segment_ids = tf.zeros((1, 384))

model.predict([input_ids, input_mask, segment_ids])
