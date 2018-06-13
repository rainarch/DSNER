# -*- coding: utf-8 -*-

class Config():
    def __init__(self):
        # dataset path for training, dev and testing, two datasets used in our experiments are saved in the folder 'data'
        self.dataset = {
            'traindata': r'data/EC/train',
            'devdata': r'data/EC/dev',
            'testdata': r'data/EC/test',
        }
        '''
            distantly supervised data path; 
            if you don't want to use ds data like the second baseline experiment, you MUST set this variable as None!!!
        '''
        self.DS_data = r'data/EC/ds_fa'  
        #self.DS_data = None

        self.map_dict = {             # the path to save mapping resources
            'char2id': r'resource/mapping/char2id',
            'label2id': r'resource/mapping/label2id',
        }

        '''
             the length of each input sentence
                100 for MSRA data
                75 for EC data
        '''
        self.maxlen = 75 

		# save and reload path
	    self.modelpath = r'Model/best_model.ckpt'
        self.modeldir = r'Model/'

        '''
            parameters for training model
                batch_size: 128 for MSRA data
                            64 for EC data
        '''
        self.model_para = {
            'lr': 0.001,
            'dropout_rate': 0.2,
            'batch_size': 64,
            'lstm_layer_num': 1,
            'input_dim': 100,
            'hidden_dim': 100,
            'emb_path': None,  # path of pre-train embeddings
        }

        # the epoch number for training
        self.epochs = 800

