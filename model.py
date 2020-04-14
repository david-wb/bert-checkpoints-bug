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
