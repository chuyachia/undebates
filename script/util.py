import numpy as np
import csv
import collections
from nltk import tokenize
import re
import os.path
import _pickle

class ModelHelper():
    def __init__(self,tok2id,id2tok, w_frequency):
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.w_frequency = w_frequency

    @classmethod
    def build(cls,data,size=5000):                        
        tok2id = {}
        id2tok = []
        data =  [w for s in data for w in s]
        w_frequency = [('<UNK>', -1)]
        idx= 0 
        w_frequency.extend(collections.Counter(data).most_common(size))
        for w in w_frequency:
            if w[0] not in tok2id:
                tok2id[w[0]] = idx
                id2tok.append(w[0])
                idx += 1
        return cls(tok2id, id2tok, w_frequency)
    
    def vectorize_sent(self,sentence):
        sentence_ = [self.tok2id.get(word, self.tok2id['<UNK>']) for word in sentence]
        return sentence_

    def vectorize_data(self,data):
        return [self.vectorize_sent(sentence) for sentence in data]
	
def load_and_preprocess_data(data_path,country_code):
    # load data
    text_str = ''
    with open(data_path,'r', encoding='utf-8',newline='') as file:
        read = csv.DictReader(file, delimiter=',', quotechar='"')
        for row in read:
            if row['country'] == country_code:
                text_str+=' '+row['text']
    # preprocess data 
    # output : [[w,w,w,w],[w,w,w,w,w],...]
    sent_str = tokenize.sent_tokenize(text_str)
    sent_arr = []
    for s in sent_str:
        words = re.sub('(?<=\w)([!?.,;\'])', r' \1',s).replace('\ufeff',' ').replace('\n',' ').lower().split()
        words.append('<EOS>')
        sent_arr.append(words)
            
    helper = ModelHelper.build(sent_arr)
    vectorize_sent_arr = helper.vectorize_data(sent_arr)
	
    return helper, vectorize_sent_arr
	
def load_embeddings(country_code):
    if os.path.exists("saved_params/saved_embedding_{}.npy".format(country_code)):
        with open("saved_params/saved_embedding_{}.npy".format(country_code) , "rb") as f:
            embedding =_pickle.load(f)
        print('Loaded pre-trained word embeddings')
        return embedding
    else:
        return None


def get_random_context(data,batch_size,windows_size,number_skip):
    center = []
    target = []
    for _ in range(batch_size//number_skip):
        sent = data[np.random.randint(0,len(data))] # pick a sentence
        while windows_size>=len(sent)-windows_size:
            sent = data[np.random.randint(0,len(data))] 
        exclude_indx = []
        center_indx = np.random.randint(windows_size,len(sent)-windows_size) # pick a center word that allows full window
        for _ in range(number_skip):
            span = list(range(center_indx-windows_size,center_indx+windows_size+1))
            exclude_indx.append(center_indx)
            target_indx = center_indx
            while target_indx in exclude_indx:
                target_indx = np.random.choice(span)                
            center.append(sent[center_indx])
            target.append(sent[target_indx])			
    center = np.array(center)
    target = np.reshape(np.array(target),(-1,1))
    return center,target
	
def get_batches(inputs,labels, seq_len):
    assert inputs.shape == labels.shape
    
    num_batches = inputs.shape[1]//seq_len
    for batch_indx in range(num_batches):
        start_indx = batch_indx*seq_len
        end_indx = start_indx+seq_len
        inputs_batch = inputs[:,start_indx:end_indx]
        labels_batch = labels[:,start_indx:end_indx]
        yield inputs_batch,labels_batch

def choose_words(n,probs):
    p = np.squeeze(probs) 
    p[np.argsort(p)[:-n]] = 0 
    p = p / np.sum(p) 
    return np.random.choice(len(p), 1, p=p)[0]

def top_n_word(n,probs):
    probs = np.squeeze(probs)
    top_indx= np.argsort(probs)[-n:]
    if 0 in top_indx: # token 0 = 'UNK'
        top_indx[np.where(top_indx ==0)] =  np.argsort(probs)[-(n+1)]
    top_prob = probs[top_indx]    
    return top_indx, top_prob 
