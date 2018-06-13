import codecs
import os
import math
import utils
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from Config import Config


class LSTM_CRF_PA(object):
    def __init__(self, num_chars, num_classes, num_steps=200, num_epochs=100, embedding_matrix=None, is_crf=True, weight=False):
        # Parameter
        self.max_f1 = 0
        self.overbest = 0
        self.config = Config()
        self.learning_rate = self.config.model_para['lr']
        self.dropout_rate = self.config.model_para['dropout_rate']
        self.batch_size = self.config.model_para['batch_size']
        self.num_layers = self.config.model_para['lstm_layer_num']
        self.input_dim = self.config.model_para['input_dim']
        self.hidden_dim = self.config.model_para['hidden_dim']
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_classes = num_classes

        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets_weight = tf.placeholder(tf.float32, [None, self.num_steps])
        self.targets_transition = tf.placeholder(tf.int32, [None])

        self.PA_targets = tf.placeholder(tf.int32, [None, self.num_steps+2, self.num_classes+1])

        # char embedding
        if embedding_matrix is not None:
            self.embedding = tf.Variable(embedding_matrix, trainable=True, name="char_emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("char_emb", [self.num_chars, self.input_dim])
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.input_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)


        # lstm cell
        lstm_cell_fw = rnn.LSTMCell(self.hidden_dim)
        lstm_cell_bw = rnn.LSTMCell(self.hidden_dim)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)

        #if is_training:
        lstm_cell_fw = rnn.DropoutWrapper(lstm_cell_fw, input_keep_prob = self.keep_prob, output_keep_prob = self.keep_prob)
        lstm_cell_bw = rnn.DropoutWrapper(lstm_cell_bw, input_keep_prob = self.keep_prob, output_keep_prob = self.keep_prob)

        
        lstm_cell_fw = rnn.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = rnn.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32) 
        
        # forward and backward
        self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            lstm_cell_fw, 
            lstm_cell_bw,
            self.inputs_emb, 
            dtype=tf.float32,
            sequence_length=self.length,
        )

        # softmax
        self.outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, self.hidden_dim * 2])        
        self.outputs = tf.nn.dropout(self.outputs, keep_prob = self.keep_prob)
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        if not is_crf:
            pass
        else:
            self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])
            self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])
            dummy_val = -1000
            class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
            self.observations = tf.concat([self.tags_scores, class_pad], 2)
            
            begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32)
            end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32) 
            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])
            
            self.observations = tf.concat([begin_vec, self.observations, end_vec], 1)
            
            self.mask = tf.cast(tf.reshape(tf.sign(self.targets),[self.batch_size * self.num_steps]), tf.float32)
            
            
            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]), tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(self.targets,[self.batch_size * self.num_steps]))
            self.point_score *= self.mask

            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)
        
            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)  

            # new PA scores
            self.PA_path_score = self.PA_forward(self.observations, self.transitions, self.length, self.PA_targets)

            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre  = self.forward(self.observations, self.transitions, self.length)
            # loss
            self.loss_PA = - (self.PA_path_score - self.total_path_score)
            
        self.optimizer_PA = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss_PA) 


    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def logsumexp_PA(self, x, pre_real_tags, islast=False):
        result = []
        for i in range(self.batch_size):
            x_max = tf.reduce_max(x[i,:,:], reduction_indices=0, keep_dims=True)
            x_max_ = tf.reduce_max(x[i,:,:], reduction_indices=0)
            b_select = tf.cast(pre_real_tags[i], dtype=tf.bool)
            select = tf.where(b_select)
            select = tf.reshape(select,[-1])
            sample = tf.gather(x[i,:,:], select)
            result.append(x_max_ + tf.log(tf.reduce_sum(tf.exp(sample - x_max), reduction_indices=0)))
        
        if islast==False:
            result = tf.reshape(tf.concat(result, 0),[self.batch_size, self.num_classes+1])
        else:
            result = tf.reshape(tf.concat(result, 0),[self.batch_size])
        return result
        
        

    def PA_forward(self, observations, transitions, length, y_PA_batch):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, self.num_classes+1, self.num_classes+1])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, self.num_classes+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])

        previous = observations[0, :, :, :]
        alphas = [previous]
        pre_real_tags = []
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_classes+1])
            alpha_t = previous + current + transitions

            # collect pre_real_tags
            for ba in range(self.batch_size):
                pre_real_tags.append(y_PA_batch[ba][t-1])

            alpha_t = tf.reshape(self.logsumexp_PA(alpha_t, pre_real_tags), [self.batch_size, self.num_classes+1, 1])
            alphas.append(alpha_t)
            previous = alpha_t
            pre_real_tags = []

        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, self.num_classes+1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), self.num_classes+1, 1])
        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_classes+1, 1])
        
        for ba in range(self.batch_size):
            pre_real_tags.append(y_PA_batch[ba][length[ba]])
        return tf.reduce_sum(self.logsumexp_PA(last_alphas, pre_real_tags, islast=True))



    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, self.num_classes+1, self.num_classes+1])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, self.num_classes+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]

        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_classes+1])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, self.num_classes+1, 1])
            alphas.append(alpha_t)
            previous = alpha_t
            

        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, self.num_classes+1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), self.num_classes+1, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_classes+1, 1])

        max_scores = tf.reshape(tf.concat(max_scores, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre

    
    def getTransition(self, y_train_batch):
        transition_batch = []
        for m in range(len(y_train_batch)):
            y = [self.num_classes] + list(y_train_batch[m]) + [0]
            for t in range(len(y)):
                if t + 1 == len(y):
                    continue
                i = y[t]
                j = y[t + 1]
                if j == 0:
                    break
                transition_batch.append(i * (self.num_classes+1) + j)
        transition_batch = np.array(transition_batch)
        return transition_batch


    def train(self, sess, X_char_merge_train, y_merge_train, X_char_dev, y_dev, X_char_test, y_test):
        char2id, id2char = utils.loadMap(self.config.map_dict['char2id'])
        label2id, id2label = utils.loadMap(self.config.map_dict['label2id'])

        merged =  tf.summary.merge_all()

        num_iterations = int(math.ceil(1.0 * len(X_char_merge_train) / self.batch_size))
        for epoch in range(self.num_epochs):
            print "current epoch: %d" % (epoch)
            cnt = 0
            for iteration in range(num_iterations):
                # get PA data
                X_char_PA_batch, y_PA_batch = utils.nextBatch(X_char_merge_train, y_merge_train, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                _, loss_PA, length=\
                    sess.run([
                        self.optimizer_PA,
                        self.loss_PA,
                        self.length,
                    ],
                    feed_dict={
                        self.inputs:X_char_PA_batch,
                        self.keep_prob:1-self.dropout_rate,
                        self.PA_targets:y_PA_batch
                    })

            if iteration == num_iterations-1:
                cnt += 1
                print "epoch: %d, train_PA loss: %5d" % (epoch ,loss_PA)
                self.test(sess, X_char_dev, y_dev, epoch, cnt, istest = False)
                self.test(sess, X_char_test, y_test, epoch, cnt, istest = True)


    def viterbi(self, max_scores, max_scores_pre, length, predict_size=128):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths
    
    def evaluate(self, y_true, y_pred, id2char, id2label, epoch, cnt, istest):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        eval_script = 'tmp/conlleval'
        output_path = 'tmp/evaluate.txt'
        scores_path = 'tmp/score.txt'
        
        with open(output_path,'w')as f:
            for i in range(len(y_true)):
                for j in range(len(y_true[i])):
                    if y_true[i][j] not in id2label:
                        print y_true[i][j], i, j
                    if id2label[y_true[i][j]]=='<PAD>':
                        break
                    f.write(id2label[y_true[i][j]]+' '+id2label[y_true[i][j]]+' '+id2label[y_true[i][j]]+' '+id2label[y_pred[i][j]]+'\n')
                f.write('\n')
        
        os.system("perl %s < %s > %s" % (eval_script, output_path, scores_path))
        eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
        if istest:
            score_test = 'tmp/score_test'
        else:
            score_test = 'tmp/score_dev'
        with open(score_test,'a')as fw:
            fw.write('epoch:  '+str(epoch)+'  '+'iteration:  '+str(cnt)+'  '+eval_lines[1]+'\n')
        if self.overbest == 1:
            self.overbest = 0
            with open('tmp/best_score','a')as fw2:
                fw2.write('epoch:  '+str(epoch)+'  '+'iteration:  '+str(cnt)+'  dev:  '+str(self.max_f1)+'  test:  '+eval_lines[1].split("FB1:")[-1].strip()+'\n')
        return eval_lines[1]


    def test(self, sess, X_char_test, y_test, epoch, cnt, istest = False):
        char2id, id2char = utils.loadMap(self.config.map_dict['char2id'])
        label2id, id2label = utils.loadMap(self.config.map_dict['label2id'])
        num_iterations = int(math.ceil(1.0 * len(X_char_test) / self.batch_size))
        preds = []
        for i in range(num_iterations):
            
            X_char_test_batch = X_char_test[i * self.batch_size : (i + 1) * self.batch_size]
            if i == num_iterations - 1 and len(X_char_test_batch) < self.batch_size:
                X_char_test_batch = list(X_char_test_batch)
                last_size = len(X_char_test_batch)
                X_char_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                X_char_test_batch = np.array(X_char_test_batch)
                length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores, self.max_scores_pre], feed_dict={self.inputs:X_char_test_batch, self.keep_prob:1})
                predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)        
                preds.extend(predicts[:last_size])
            else:
                X_char_test_batch = np.array(X_char_test_batch)
                length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores, self.max_scores_pre], feed_dict={self.inputs:X_char_test_batch, self.keep_prob:1})
                predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
                preds.extend(predicts)
        
        result = self.evaluate(y_test, preds, id2char, id2label, epoch, cnt, istest)

        if float(result.split("FB1:")[-1].strip()) > self.max_f1 and not istest:
            #saver = tf.train.Saver()
            self.overbest = 1
            self.max_f1 = float(result.split("FB1:")[-1].strip())
            #save_path = saver.save(sess, self.config.modelpath, global_step = epoch)
            print "saved the best model with f1:  ", self.max_f1

