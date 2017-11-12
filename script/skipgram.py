# try visualize trained embedding
import numpy as np
import tensorflow as tf
import util
import _pickle
import viz

class Config:
    neg_samp = 128
    n_batches = 100000
    batch_size = 132
    embed_size = 128
    windows_size = 3
    number_skip = 2
    lr = 0.001
    def __init__(self,tokens,country_code): # takes vocabulary size from here
        self.vocab_size = len(tokens)
        self.country_code = country_code
		
class SkipgramModel(object):
    def add_placeholders(self):
        self.center_placeholder = tf.placeholder(tf.int32,shape=[self.config.batch_size])
        self.target_placeholder = tf.placeholder(tf.int32,shape = [self.config.batch_size,1])
            
    def create_feed_dict(self,center_batch,target_batch):
        feed_dict = {
            self.center_placeholder:center_batch,
            self.target_placeholder:target_batch
            }
        return feed_dict
            
    def add_embed_op(self):	
        embed_matrix = tf.get_variable(name = 'Embed', shape = [self.config.vocab_size,self.config.embed_size],initializer = tf.contrib.layers.xavier_initializer())
        return embed_matrix
            
    def add_loss_op(self,embed_matrix):
        embedding = tf.nn.embedding_lookup(embed_matrix,self.center_placeholder)
        weights = tf.get_variable(name = 'W', shape = [self.config.vocab_size,self.config.embed_size],initializer= tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name = 'b', initializer=  tf.zeros(shape=[self.config.vocab_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights = weights, biases = biases, labels = self.target_placeholder, inputs = embedding, num_sampled = self.config.neg_samp, num_classes = self.config.vocab_size))
        return loss
            
    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op
        
    def train(self, sess, center_batch, target_batch):
        feed = self.create_feed_dict(center_batch=center_batch,target_batch=target_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss
    
    def fit(self, sess, data, save):
        for b in range(self.config.n_batches+1):
            center_batch, target_batch= util.get_random_context(data=data, batch_size= self.config.batch_size,
                                                                windows_size= self.config.windows_size, number_skip= self.config.number_skip)
            loss = self.train(sess, center_batch=center_batch,target_batch=target_batch)
            if b%10000 ==0:
                print('Number of batch: {} Loss: {}'.format(b,loss))
                    
    def get_trained_embedding(self,sess):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embed_matrix), 1, keep_dims=True))
        embed_matrix_norm  = self.embed_matrix / norm
        trained_embedding = sess.run(embed_matrix_norm)
        return trained_embedding
            
    def build(self):
        self.add_placeholders()
        self.embed_matrix = self.add_embed_op()
        self.loss = self.add_loss_op(self.embed_matrix)
        self.train_op = self.add_training_op(self.loss)
    
    def __init__(self,config):
        self.config = config
        self.center_placeholder  = None
        self.target_placeholder = None
        self.build()
				
def do_train(country_code,helper=None, data_raw= None, save= True,draw=True):
    if not helper or not data_raw:
        helper, data_raw = util.load_and_preprocess_data(data_path='../input/un-general-debates.csv',country_code=country_code)
    config = Config(helper.tok2id,country_code)
    model = SkipgramModel(config)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        model.fit(sess,data_raw,save=True)
        trained_embedding = model.get_trained_embedding(sess)
        if save:
            with open("saved_params/saved_embedding_{}.npy".format(model.config.country_code) , "wb") as f:
                _pickle.dump(trained_embedding, f)
        if draw:
            word_to_plot, word_frequency = viz.frequent_non_stop_word(count= helper.w_frequency,number=150)
            viz.embedding(tokens=helper.tok2id,embedding=trained_embedding,wordlist=word_to_plot,wordsize=word_frequency,
                          filename='../output/embed_{}.png'.format(country_code),title='{}-150 most frequent non stop words'.format(country_code))
        return trained_embedding
        
    
if __name__ =='__main__':
    do_train('FRA')
                                       
