import os
import time
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from LSTM_CRF_PA import LSTM_CRF_PA
from Config import Config

print "preparing data"

con=Config()

(X_char_train, y_train), (X_char_dev, y_dev), (X_char_test, y_test), (X_char_PA, y_PA)  = utils.get_AllData(con.maxlen)

char2id, id2char = utils.loadMap(con.map_dict['char2id'])
label2id, id2label = utils.loadMap(con.map_dict['label2id'])

num_chars = len(id2char.keys())
num_classes = len(id2label.keys())

print 'num of chars:', num_chars
print 'num of classes:', num_classes

#X_char_train, y_train = utils.get_PA_data(con.dataset['traindata'], char2id, label2id)

# merge export and PA train data
X_char_merge_train, y_merge_train = utils.merge_export_and_PA_train_data(X_char_train, y_train, X_char_PA, y_PA)

if con.model_para['emb_path'] != None:
    embedding_matrix = utils.getEmbedding(con.model_para['emb_path'], char2id)
else:
    embedding_matrix = None

print "building model"

num_steps=con.maxlen
num_epochs=con.epochs


cpu_config = tf.ConfigProto(device_count={"CPU": 8}, # limit to num_cpu_core CPU usage
    inter_op_parallelism_threads = 8,
    intra_op_parallelism_threads = 8,
    allow_soft_placement=True,
)


with tf.Session(config=cpu_config) as sess:
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = LSTM_CRF_PA(num_chars=num_chars, num_classes=num_classes, num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix)
    print "training model"
    sess.run(tf.global_variables_initializer())
    model.train(sess, X_char_merge_train, y_merge_train, X_char_dev, y_dev, X_char_test, y_test)


