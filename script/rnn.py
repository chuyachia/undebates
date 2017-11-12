import numpy as np
import tensorflow as tf
import util
import os.path
from datetime import datetime

class Config:
    state_size = 512
    n_layers = 3
    #loss_samp = 128 if use nce loss or sampled softmax
    n_epochs = 50
    seq_len = 20
    batch_size = 50
    lr = 0.001
    #max_grad = 5.0 # avoid exploding gradient
    def __init__(self,tokens,country_code): # takes vocabulary size from here
        self.vocab_size = len(tokens)
        self.country_code = country_code
            
class RNNModel(object):
    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32,[None,None]) #[batch_size,seq_len]
        self.labels_placeholder = tf.placeholder(tf.int32,[None,None]) 
        self.state_placeholder = tf.placeholder(tf.float32,[None,self.config.state_size*self.config.n_layers])
		
    def create_feed_dict(self,inputs_batch, state, labels_batch=None):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.state_placeholder:state
            }
			
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        embed_matrix = tf.Variable(self.pretrained_embeddings,dtype=tf.float32,name='embed_matrix')
        embeddings = tf.nn.embedding_lookup(embed_matrix,self.inputs_placeholder)
        embeddings = tf.cast(embeddings, tf.float32)
        return embeddings
    
    def add_prediction_op(self):
        x = self.add_embedding()
        # define self.config.keep_prob first if to use drop out
        #dropcells = [tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=self.config.keep_prob) for cell in cells] 
        #multicell = tf.nn.rnn_cell.MultiRNNCell(dropcells, state_is_tuple=False)
        #multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, output_keep_prob=self.config.keep_prob)
        cells = [tf.nn.rnn_cell.GRUCell(self.config.state_size) for _ in range(self.config.n_layers)]
        multicell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)
        outputs, state = tf.nn.dynamic_rnn(multicell,x,dtype=tf.float32)
        W2 = tf.get_variable(name = 'W2', shape = [self.config.state_size,self.config.vocab_size],initializer= tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name = 'b2', initializer=  tf.zeros(shape=[1,self.config.vocab_size]))
        preds = tf.map_fn(lambda x: tf.matmul(x, W2) + b2 , outputs)
        preds = tf.reshape(preds, [-1, self.config.vocab_size])
        return preds, state
        
    def add_loss_op(self, preds):
        labels_series = tf.reshape(self.labels_placeholder, [-1])
        losses_temp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels_series,logits= preds)
        loss = tf.reduce_mean(losses_temp)
        return loss
    
    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op
    
    def train(self, sess, inputs_batch, labels_batch, init_state):
        feed = self.create_feed_dict(inputs_batch, state= init_state, labels_batch=labels_batch)
        _, loss, state = sess.run([self.train_op, self.loss, self.state], feed_dict=feed)
        return loss, state

    def evaluate(self, sess,inputs_batch, labels_batch, init_state):
        feed = self.create_feed_dict(inputs_batch, state= init_state, labels_batch=labels_batch)
        loss = sess.run(self.loss, feed_dict=feed)
        return loss
	
    def predict(self, sess,inputs_batch, init_state):
        feed = self.create_feed_dict(inputs_batch, state= init_state)
        preds,state = sess.run([self.preds,self.state], feed_dict=feed)
        return preds,state
		
    def preprocess_sequence_data(self,data):
        data_concat = np.array([word for sent in data for word in sent])
        num_batches = len(data_concat)//self.config.batch_size//self.config.seq_len
        x = data_concat[:num_batches*self.config.batch_size*self.config.seq_len]
        y = np.roll(x,-1)
        x = np.reshape(x, (self.config.batch_size,-1))
        y = np.reshape(y, (self.config.batch_size,-1))
        return x,y
        # the model is only intended to learn the discoure, no need of evaluation
        #eval_x = x[-self.config.batch_size*self.config.seq_len:] #leave examples of the size one batch for evaluation
        #train_x = x[:-self.config.batch_size*self.config.seq_len]
        #eval_y = y[-self.config.batch_size*self.config.seq_len:]
        #train_y = y[:-self.config.batch_size*self.config.seq_len]
        #eval_x = np.reshape(eval_x, (self.config.batch_size,-1))
        #train_x = np.reshape(train_x, (self.config.batch_size,-1))
        #eval_y = np.reshape(eval_y, (self.config.batch_size,-1))
        #train_y = np.reshape(train_y, (self.config.batch_size,-1))
        #return train_x,train_y,eval_x,eval_y #[batch_size,seq_len*number_batches]
			
		
    def fit(self, sess, saver,data):
        best_eval_loss = 999		
        x,y = self.preprocess_sequence_data(data)
        print('Start fitting')
        for epoch in range(self.config.n_epochs+1): 
            train_loss = []
            state = np.zeros((self.config.batch_size, self.config.state_size*self.config.n_layers))
            for inputs_batch,labels_batch in util.get_batches(x,y, self.config.seq_len):
                loss, state = self.train(sess, inputs_batch,labels_batch,state)
                #print('Batch loss: {}'.format(loss))
                train_loss.append(loss)
            train_loss = sum(train_loss)/len(train_loss)
            print('Epoch: {0} Training loss {1:.2f}'.format(epoch,train_loss))
            if epoch%5 ==0:
                if not os.path.exists('rnn_check_points/{}'.format(self.config.country_code)):
                    os.makedirs('rnn_check_points/{}'.format(self.config.country_code))
                saver.save(sess, 'rnn_check_points/{}/rnn'.format(self.config.country_code),global_step=epoch)
                print('Parameters saved.')
                #eval_loss = []
                #for inputs_batch,labels_batch in util.get_batches(eval_x,eval_y,self.config.seq_len):
                #    loss = self.evaluate(sess, inputs_batch,labels_batch,state)
                #    eval_loss.append(loss)
                #eval_loss = sum(eval_loss)/len(eval_loss)
                #print('Epoch: {0} Evaluation loss {1:.2f}'.format(epoch,eval_loss))
                #if eval_loss < best_eval_loss:
                #    if not os.path.exists('rnn_check_points/{}'.format(self.config.country_code)):
                #        os.makedirs('rnn_check_points/{}'.format(self.config.country_code))
                #    saver.save(sess, 'rnn_check_points/{}/rnn'.format(self.config.country_code),global_step=epoch)
                #    print('Best evaluation loss. Parameters saved.')
                #    best_eval_loss = eval_loss
		
    def random_search(self, sess, input_word,state,top_n):
        preds, state = self.predict(sess,input_word,state)
        preds = np.exp(preds) / np.sum(np.exp(preds))
        w_id = 0
        while w_id == 0:
            w_id = util.choose_words(top_n,preds)
        return w_id,state

    def build(self):
        self.add_placeholders()
        self.preds, self.state = self.add_prediction_op()
        self.loss = self.add_loss_op(self.preds)
        self.train_op = self.add_training_op(self.loss)
        
    def __init__(self,config,pretrained_embeddings):
        self.config = config
        self.pretrained_embeddings = pretrained_embeddings
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.build()

def do_train(country_code,embedding, helper, data_raw):
    config = Config(helper.tok2id,country_code)
    with tf.Graph().as_default():
        model = RNNModel(config,embedding)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep= 1)
        with tf.Session() as sess:
            sess.run(init)
            model.fit(sess,saver,data_raw)

def generate_w_random_search(country_code,embedding,helper,prime_word,n_sents,top_n):
    config = Config(helper.tok2id,country_code)
    with tf.Graph().as_default():
        model = RNNModel(config,embedding)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('rnn_check_points/{}'.format(country_code))
            saver.restore(sess, ckpt.model_checkpoint_path)
            w_id = np.array([[helper.tok2id[prime_word]]])
            print('Start generating')
            print('"',end=' ')
            print(prime_word, end=' ')
            state =np.zeros((1, model.config.state_size*model.config.n_layers))
            sent = 0
            with open('../output/generated_{}_{:%Y%m%d_%H%M%S}.txt'.format(country_code, datetime.now()),'w') as f:
                f.write(prime_word+' ')
                while sent < n_sents:
                    w_id, state = model.random_search(sess,w_id, state,top_n)
                    w_char = helper.id2tok[w_id]
                    print(w_char, end = ' ')
                    if w_char == '<EOS>':
                        f.write('\n')
                        sent+=1
                    else:
                        f.write(w_char+' ')
                    w_id = np.array([[w_id]])
                print('"')
            print('Generated text saved to output')            



if __name__ =="__main__":
    country_code ='FRA'
    helper, data_raw = util.load_and_preprocess_data(data_path='../input/un-general-debates.csv',country_code=country_code)
    embedding = util.load_embeddings(country_code)
    assert embedding is not None, 'No pretrained embeddings found. Use skipgram.py to train word embeddings'
    #do_train(country_code,embedding,helper, data_raw)
    generate_w_random_search(country_code,embedding, helper, 'palestine',20,5)
  
            
        
