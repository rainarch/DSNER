# -*- coding: UTF-8 -*-
import re
import os
import types
import csv
import time
import pickle
import numpy as np
import pandas as pd
import random
from Config import Config
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

np.random.seed(1337)
config=Config()

def getEmbedding(emb_file, char2id):
    emb_dic={}
    with open(emb_file,'rb')as f:
        for line in f.readlines():
            line=line.rstrip().decode('utf-8')
            line_list=line.split('\t')
            key=line_list[0]
            line_list.pop(0)
            for i in xrange(len(line_list)):
                line_list[i] = float(line_list[i])
            emb_dic[key]=line_list
    embedding_matrix = np.zeros((len(char2id.keys()),config.model_para['input_dim']))
    count=0
    for key in char2id.keys():
        embedding_vector=emb_dic.get(key)
        if embedding_vector is not None:
            count+=1
            embedding_matrix[char2id[key]] = embedding_vector
    print 'get_emb_count:',count
    return embedding_matrix




def saveMap(id2char, id2label):
    with open(config.map_dict['char2id'], "wb") as outfile:
        for idx in id2char:
            outfile.write(id2char[idx].encode('utf-8') + "\t" + str(idx)  + "\r\n")

    with open(config.map_dict['label2id'], "wb") as outfile:
        for idx in id2label:
            outfile.write(id2label[idx].encode('utf-8') + "\t" + str(idx) + "\r\n")
    print "saved map between token and id"



def get_resource_list(path):
    df_train = pd.read_csv(path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"], encoding='utf-8')
    char = list(set(df_train["char"][df_train["char"].notnull()]))
    label = list(set(df_train["label"][df_train["label"].notnull()]))
    return char, label

def buildMap():
    
    char=[]
    label=[]
    for path in config.dataset.values():
        c, l = get_resource_list(path)
        char.extend(c)
        label.extend(l)
    
    # map char of PA_data
    if config.DS_data is not None:
        df_train_PA = pd.read_csv(config.DS_data, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"], encoding='utf-8')
        char_PA = list(set(df_train_PA["char"][df_train_PA["char"].notnull()]))
        char.extend(char_PA)

    char = list(set(char))
    label = list(set(label))

    char2id = dict(zip(char, range(1, len(char) + 1)))
    label2id = dict(zip(label, range(1, len(label) + 1)))
    id2char = dict(zip(range(1, len(char) + 1), char))
    id2label =  dict(zip(range(1, len(label) + 1), label))

    id2char[0] = "<PAD>"
    id2label[0] = "<PAD>"
    char2id["<PAD>"] = 0
    label2id["<PAD>"] = 0

    id2char[len(char) + 1] = "<NEW>"
    char2id["<NEW>"] = len(char) + 1
	
    saveMap(id2char, id2label)

    return char2id, id2char, label2id, id2label


# use "0" to padding the sentence
def padding(sample, maxlen):
    for i in range(len(sample)):
        if len(sample[i]) < maxlen:
            sample[i] += [0 for _ in range(maxlen - len(sample[i]))]
    return sample


def prepare(chars, labels, maxlen, is_padding=True):
    X_char = []
    y = []
    tmp_char = []
    tmp_y = []
    for record in zip(chars, labels):
        c = record[0]
        l = record[1]
        # empty line
        if c == -1:
            if len(tmp_char) <= maxlen:
                X_char.append(tmp_char)
                y.append(tmp_y)
            tmp_char = []
            tmp_y=[]
        else:
            tmp_char.append(c)
            tmp_y.append(l)
    if is_padding:
        X_char = np.array(padding(X_char, maxlen))
    else:
        X_char = np.array(X_char)
    y = np.array(padding(y, maxlen))
    
    return X_char, y


def get_data(path, char2id, label2id):
    df_train = pd.read_csv(path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"], encoding='utf-8')

    # map the char , label into id
    df_train["char_id"] = df_train.char.map(lambda x : -1 if str(x) == str(np.nan) else char2id[x])
    df_train["label_id"] = df_train.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])
    
    X_char, y = prepare(df_train["char_id"], df_train["label_id"], config.maxlen)
    return (X_char, y)

def get_PA_data(path, char2id, label2id):
    if path is None: return [], []
    df_train = pd.read_csv(path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["char", "label"], encoding='utf-8')
    df_train["char_id"] = df_train.char.map(lambda x : -1 if str(x) == str(np.nan) else char2id[x])
    sentence_pa = []
    sentence=[]
    padding_hot = [0 for i in range(len(label2id)+1)]
    padding_hot[0] = 1
    head_hot = [0 for i in range(len(label2id)+1)]
    head_hot[len(label2id)] = 1
    for i in range(len(df_train.label)):
        if str(df_train.label[i]) == str(np.nan):
            for j in range(config.maxlen-len(sentence)):
                sentence.append(padding_hot)
            sentence.insert(0, head_hot)
            sentence.append(padding_hot)
            sentence_pa.append(sentence)
            sentence=[]
        else:
            if df_train.label[i]==u'UNK':
                sentence.append([1 for i in range(len(label2id)+1)])
            else:
                index = int(label2id[df_train.label[i]])
                label_pa=[0 for i in range(len(label2id)+1)]
                label_pa[index]=1
                sentence.append(label_pa)

    X_char, y = prepare(df_train["char_id"], df_train["char_id"], config.maxlen)
    return X_char, sentence_pa



def get_AllData(maxlen):
    char2id, id2char, label2id, id2label = buildMap()
  
    return get_PA_data(config.dataset['traindata'], char2id, label2id),get_data(config.dataset['devdata'], char2id, label2id), get_data(config.dataset['testdata'], char2id, label2id), get_PA_data(config.DS_data, char2id, label2id)


def merge_export_and_PA_train_data_with_sign(X_char_train, y_train, X_char_PA, y_PA):
    sample = []
    all_sample = []
    expert_sign = [0]
    PA_sign = [1]
    for i in range(len(X_char_train)):
        sample.append(X_char_train[i])
        sample.append(y_train[i])
	# append a sign to show hand_tagged_data :[0]
	sample.append(expert_sign)
        all_sample.append(sample)
        sample=[]
    sample=[]
    for i in range(len(X_char_PA)):
        sample.append(X_char_PA[i])
        sample.append(y_PA[i])
	# append a sign to show PA_data :[1]
	sample.append(PA_sign)
        all_sample.append(sample)
        sample=[]
    
    random.shuffle(all_sample)
    X_char_merge_train = []
    y_merge_train = []
    sign_merge_train = []
    for i in range(len(all_sample)):
        X_char_merge_train.append(all_sample[i][0])
        y_merge_train.append(all_sample[i][1])
	sign_merge_train.append(all_sample[i][2][0])
    return X_char_merge_train, y_merge_train, sign_merge_train

def merge_export_and_PA_train_data(X_char_train, y_train, X_char_PA, y_PA):
    sample = []
    all_sample = []
    for i in range(len(X_char_train)):
        sample.append(X_char_train[i])
        sample.append(y_train[i])
        all_sample.append(sample)
        sample=[]
    sample=[]
    for i in range(len(X_char_PA)):
        sample.append(X_char_PA[i])
        sample.append(y_PA[i])
        all_sample.append(sample)
        sample=[]
    
    random.shuffle(all_sample)
    X_char_merge_train = []
    y_merge_train = []
    for i in range(len(all_sample)):
        X_char_merge_train.append(all_sample[i][0])
        y_merge_train.append(all_sample[i][1])

    return X_char_merge_train, y_merge_train


def mapFunc(x, char2id):
    if str(x) == str(np.nan):
        return -1
    elif x not in char2id:
        return char2id["<NEW>"]
    else:
        return char2id[x]

def loadMap(token2id_filepath):
    if not os.path.isfile(token2id_filepath):
        print "file not exist, building map"
        buildMap()
    token2id = {}
    id2token = {}
    with open(token2id_filepath) as infile:
        for row in infile:
            row = row.rstrip().decode("utf-8")
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token


def nextBatch(X_char, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_char_batch = list(X_char[start_index:min(last_index, len(X_char))])
    y_batch = list(y[start_index:min(last_index, len(X_char))])
    if last_index > len(X_char):
        left_size = last_index - (len(X_char))
        for i in range(left_size):
            index = np.random.randint(len(X_char))
            X_char_batch.append(X_char[index])
            y_batch.append(y[index])
    X_char_batch = np.array(X_char_batch)
    y_batch = np.array(y_batch)
    return X_char_batch, y_batch

def reward_nextBatch(X_char, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_char_batch = list(X_char[start_index:min(last_index, len(X_char))])
    y_batch = list(y[start_index:min(last_index, len(X_char))])
    if last_index > len(X_char):
        left_size = last_index - (len(X_char))
        for i in range(left_size):
            X_char_batch.append(X_char[i])
            y_batch.append(y[i])
    X_char_batch = np.array(X_char_batch)
    y_batch = np.array(y_batch)
    return X_char_batch, y_batch







