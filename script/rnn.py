import numpy as np
import tensorflow as tf
import util
import os.path
from datetime import datetime


class Config:
    state_size = 200 # mean of input size and output size?
    n_layers = 1
    seq_len = 10
    batch_size = 400
    lr = 1e-4
    max_grad = 5.0
    def __init__(self,tokens,country_code): 
        self.vocab_size = len(tokens)
        self.country_code = country_code
            
class RNNModel(object):
    def setup_embeddings(self,embed_path):
        with np.load(os.path.join(embed_path,'embedding_200.npz')) as data:     
            vocab_embed_matrix = tf.Variable(data['vocab'],dtype=tf.float32,trainable=False,name='vocab_embed_matrix')
            common_embed_matrix = tf.Variable(data['common'],dtype=tf.float32,trainable=False,name='common_embed_matrix')
        return vocab_embed_matrix, common_embed_matrix
    
    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32,[None,None],"input") #[batch_size,seq_len]
        self.labels_placeholder = tf.placeholder(tf.int32,[None,None],"label") 
        self.state_placeholder = tf.placeholder(tf.float32,[None,self.config.state_size*self.config.n_layers],"state")
        self.word_in_vocab = tf.placeholder(tf.bool,[],"word_in_vocab")
		
    def create_feed_dict(self, state, word_in_vocab, inputs=None, labels_batch=None):
        feed_dict = {
            self.inputs_placeholder : inputs,
            self.state_placeholder:state,
            self.word_in_vocab : word_in_vocab
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        def f1():
            return tf.cast(tf.nn.embedding_lookup(self.vocab_embed_matrix,self.inputs_placeholder), tf.float32)
        def f2():
            return tf.cast(tf.nn.embedding_lookup(self.common_embed_matrix,self.inputs_placeholder), tf.float32)
        embeddings = tf.cond(self.word_in_vocab,f1,f2)
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
        b2 = tf.get_variable(name = 'b2', initializer=tf.zeros(shape=[1,self.config.vocab_size]))
        preds = tf.map_fn(lambda x: tf.matmul(x, W2) + b2 , outputs)
        preds = tf.reshape(preds, [-1, self.config.vocab_size])
        return preds, state
        
    def add_loss_op(self, preds):
        with tf.name_scope('LOSS'):
            labels_series = tf.reshape(self.labels_placeholder, [-1])
            losses_temp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels_series,logits= preds)
            loss = tf.reduce_mean(losses_temp)
            tf.summary.scalar('loss', loss)
            return loss
    
    def add_training_op(self, loss):
        with tf.name_scope('SGD'):
            #learning_rate = tf.train.exponential_decay(self.config.lr, self.global_step,100000, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients = tf.gradients(loss, tf.trainable_variables())
            gradients_cut,_ = tf.clip_by_global_norm(gradients,self.config.max_grad)
            gradients = list(zip(gradients, tf.trainable_variables()))
            gradients_cut = list(zip(gradients_cut,tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=gradients,global_step=self.global_step)
            for grad, var in gradients:
                tf.summary.histogram(var.name + '/gradient', grad)
            for grad, var in gradients_cut:
                tf.summary.histogram(var.name + '/gradient_cut', grad)
            return train_op
    
                
    def train(self, sess, inputs_batch, labels_batch, init_state):
        feed = self.create_feed_dict(inputs= inputs_batch, state= init_state, word_in_vocab = True, labels_batch=labels_batch)
        log_summary, _, loss, state = sess.run([self.merged, self.train_op, self.loss, self.state], feed_dict=feed)
        return loss, state,log_summary

    def evaluate(self, sess,inputs_batch, labels_batch, init_state):
        feed = self.create_feed_dict(inputs= inputs_batch, state= init_state, word_in_vocab = True, labels_batch=labels_batch)
        loss = sess.run(self.loss, feed_dict=feed)
        return loss
	
    def predict(self, sess,inputs_batch, init_state,word_in_vocab):
        feed = self.create_feed_dict(inputs= inputs_batch, state= init_state,word_in_vocab=word_in_vocab)
        preds,state = sess.run([self.preds,self.state], feed_dict=feed)
        return preds,state
		
    def preprocess_sequence_data(self,data):
        num_samples = len(data)//self.config.seq_len      
        num_batches = num_samples//self.config.batch_size
        print("{0} words tokens divided into {1} sequences. Total of {2} batches of size {3}"
              .format(len(data),num_samples,num_batches,self.config.batch_size))
        x = data[:num_batches*self.config.batch_size*self.config.seq_len]
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
			
		
    def fit(self, sess, saver,data,epochs):
        prev_loss = 999
        train_writer = tf.summary.FileWriter('script/log',
                                      sess.graph)
        x,y = self.preprocess_sequence_data(data)
        for epoch in range(epochs):
            print('Start epoch: {}'.format(epoch))        
            train_loss = []
            state = np.zeros((self.config.batch_size, self.config.state_size*self.config.n_layers))
            for inputs_batch,labels_batch in util.get_batches(x,y, self.config.seq_len):
                loss, state, log_summary = self.train(sess, inputs_batch,labels_batch,state)
                train_writer.add_summary(log_summary)
                train_loss.append(loss)
            train_loss = sum(train_loss)/len(train_loss)
            print('Epoch: {0} Training loss {1:.4f}'.format(epoch,train_loss))
            if train_loss <prev_loss:
                prev_loss = train_loss
                if not os.path.exists('script/rnn_check_points/{}'.format(self.config.country_code)):
                    os.makedirs('script/rnn_check_points/{}'.format(self.config.country_code))
                saver.save(sess, 'script/rnn_check_points/{}/rnn'.format(self.config.country_code),global_step=self.global_step)
                print('Best training loss. Parameters saved.')
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
		
    def random_search(self, sess, input_word,state,top_n,word_in_vocab):
        preds, state = self.predict(sess,input_word,state,word_in_vocab)
        preds = np.exp(preds) / np.sum(np.exp(preds))
        w_id = util.choose_words(top_n,preds)
        return w_id,state

    def build(self):
        self.add_placeholders()
        self.preds, self.state = self.add_prediction_op()
        self.loss = self.add_loss_op(self.preds)
        self.train_op = self.add_training_op(self.loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.merged = tf.summary.merge_all()
        print("Model built")
                         
    def __init__(self,config,embed_path):
        self.config = config
        self.vocab_embed_matrix, self.common_embed_matrix = self.setup_embeddings(embed_path)
        self.inputs_placeholder = None
        self.word_inputs_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.global_step = tf.Variable(0,name='global_step',trainable=False)
        self.build()
        

def do_train(country_code,embed_path, helper, data_raw, epochs, resume_training=True):
    config = Config(helper.tok2id,country_code)
    with tf.Graph().as_default():
        model = RNNModel(config,embed_path)
        saver = tf.train.Saver(max_to_keep= 1)
        with tf.Session() as sess:
            params = tf.trainable_variables()
            num_params = sum([np.prod(tf.shape(t.value()).eval()) for t in params])
            print("Number of params: {}".format(num_params))
            if resume_training and os.path.exists('script/rnn_check_points/{}/checkpoint'.format(country_code)):
                print('Start training from retrieved parameters')
                ckpt = tf.train.get_checkpoint_state('script/rnn_check_points/{}'.format(country_code))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Start training with new parameters')
                sess.run(tf.global_variables_initializer())
            model.fit(sess,saver,data_raw,epochs)

def generate_w_random_search(country_code,embed_path,helper,word_id,word, word_in_vocab, n_sents,top_n):
    if not os.path.exists('output'):
        os.makedirs('output')
    config = Config(helper.tok2id,country_code)
    with tf.Graph().as_default():
        model = RNNModel(config,embed_path)
        saver = tf.train.Saver()
        with tf.Session() as sess:          
            ckpt = tf.train.get_checkpoint_state('script/rnn_check_points/{}'.format(country_code))
            saver.restore(sess, ckpt.model_checkpoint_path)
            w_input = np.array([[word_id]]) 
            print('Start generating')
            print('"',end=' ')
            print(word, end=' ')
            state =np.zeros((1, model.config.state_size*model.config.n_layers))
            sent = 0
            with open('output/generated_{}_{:%Y%m%d_%H%M%S}.txt'.format(country_code, datetime.now()),'w') as f:
                f.write(word+' ')
                while sent < n_sents:
                    w_input, state = model.random_search(sess,w_input, state,top_n,word_in_vocab)
                    w_char = helper.id2tok[w_input]
                    print(w_char, end = ' ')
                    if w_char == '<EOS>':
                        f.write('\n')
                        sent+=1
                    else:
                        f.write(w_char+' ')
                    w_input = np.array([[w_input]])
                    word_in_vocab = True
                print('"')
            print('Generated text saved to output')            



if __name__ =="__main__":
    country_code ='FRA'
    helper, data_raw = util.load_and_preprocess_data(data_path='../input/un-general-debates.csv',country_code=country_code)
    embedding = util.load_embeddings(country_code)
    assert embedding is not None, 'No pretrained embeddings found. Use skipgram.py to train word embeddings'
    #do_train(country_code,embedding,helper, data_raw)
    generate_w_random_search(country_code,embedding, helper, 'palestine',20,5)
  
            
        
