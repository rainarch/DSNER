# DSNER

DSNER = Distantly Supervised NER

This project includes the code and data for our paper ‘‘Distantly Supervised NER with Partial Annotation Learning and Reinforcement Learning’’ at COLING-2018.



## Codes

Before operating program, you need to have:

    1. python2.7
    2. tensorflow\_CPU\_version>= 1.1.0

The network codes of baselines and our methods are put in the files LSTM\_CRF\_PA.py and LSTM\_CRF\_PA\_SL.py respectively.

You can set the parameters of models in the class file Config.py according to the brief note of each variable.

Other public functions are defined in the utils.py

## Resource

You can store your resources such as mapping-dict, pre-trained embeddings or saved_models in the folder Resource. And then set the paths in the configure file=Config.py.

## Dataset

You could find two datesets along with supplementary documents in the folder data which we used in the experiments.

Concretely, the data file train, dev and test are split from hand-tagged dataset with format:

```bash
我   O
要   O
买   O
一   O
台   O
游   B-cp
戏   I-cp
本   I-cp
```
where each character and its label are split by 'tab' in a line.

Distantly supervised data (partially matched data named ds_pa) is store as :
```bash
想   UNK
买   UNK
面   B-cp
膜   I-cp
```
where label 'UNK' means this character can't be matched by distant supervision.

And as mentioned in our paper, we can use these distantly matched data as supervised sentences (named ds_fa) by labeling those non-matched characters as 'O', which is one of our baseline experiments:
```bash
想   O
买   O
面   B-cp
膜   I-cp
```
## Pre-trained Embeddings

The pre-trained embeddings are trained by tool word2vec on one million sentences which are the user-generated text from Internet. We set the embedding dimension as 100, the minimum frequency of occurrence as 5, and the window size of 5. The embeddings file is available at .\resource\embedding\.

## Train

For training, you first need to ensure that each parameter has been correctly set:

1. For training LSTM_CRF_PA model:
```bash
python train_Model.py
```

2. For training LSTM_CRF_PA+SL model:
```bash
python train_DSNER_Model.py
```

The test results in the process of training will be saved in folder 'tmp'.

## Extensions

[//]: <> DSNER-pytorch (by Farhad @nooralahzadeh): It is a Pytorch version of our code. Please note that we have not tested this version yet. The project can be found at https://github.com/nooralahzadeh/DSNER-pytorch. Thank Farhad!  


## Cite

If you use the code or data, please cite the following paper:

[Yang et al., 2018] Yaosheng Yang, Wenliang Chen, Zhenghua Li, Zhengqiu He and Min Zhang. Distantly Supervised NER with Partial Annotation Learning and Reinforcement Learning, Proceedings of COLING2018, pp.2159–2169, Santa Fe, New Mexico, USA, August 20-26, 2018
