import numpy as np
import csv
import collections
from nltk import tokenize
import re
import os.path
import _pickle
from tqdm import *

class ModelHelper():
	def __init__(self,tok2id,id2tok, w_frequency):
		self.tok2id = tok2id
		self.id2tok = id2tok
		self.w_frequency = w_frequency
	@classmethod
	def build(cls,data,country_code):
                w_frequency = collections.Counter(data)
                id2tok = sorted(w_frequency, key=w_frequency.get, reverse=True)
                if not os.path.exists('script/data/{}_vocab.dat'.format(country_code)):
                        with open('script/data/{}_vocab.dat'.format(country_code),'wb') as file:
                            for w in id2tok:
                                try:
                                    file.write(w + "\n".encode('utf-8'))
                                except:
                                    file.write(w.encode('utf-8') + "\n".encode('utf-8'))
                        print('Vocabulary list written')
                tok2id = dict([(tok,id) for (id, tok) in enumerate(id2tok)])
                return cls(tok2id, id2tok, w_frequency)
	def vectorize_data(self,data):
		return [self.tok2id.get(word) for word in data]
	def add_common_words_dict(self,embed_path):
                with np.load(os.path.join(embed_path,'common_words_dict.npz')) as data:
                        self.cwtok2id = data['dictionary'][()]
	
def load_and_preprocess_data(data_path,save_path,country_code):
        save_path = os.path.join(save_path,"tokenized_data_{}.npz".format(country_code))
        if not os.path.exists(save_path):
                text_str = ''
                with open(data_path,'r', encoding='utf-8',newline='') as file:
                        read = csv.DictReader(file, delimiter=',', quotechar='"')
                        for row in read:
                                if row['country'] == country_code:
                                        text_str+=' '+row['text']
                sent_str = tokenize.sent_tokenize(text_str)
                sent_arr = []
                for s in sent_str:
                        words = re.sub('([¬])','',s)
                        words = re.sub('([‘”“\ufeff\n])', ' ',words)
                        words = re.sub('([’])','\'',words)
                        words = re.sub('(?<=[$€.,\(\)])(?=[^\s])', r' ', words)
                        words = re.sub('(?<=\w)([!?.,;:\)\(\'])', r' \1',words).lower().split()
                        words.append('<EOS>')
                        sent_arr.extend(words)
                # sent_arr: ['w','w','.','<EOS>','w',w'...]
                np.savez_compressed(save_path,data = sent_arr)
                print("Tokenized data saved to {}".format(save_path))
        with np.load(save_path) as data:
                helper = ModelHelper.build(data['data'],country_code)
                vectorized_data = helper.vectorize_data(data['data'])
        return helper, vectorized_data
                               

def process_glove(glove_dim, glove_dir, helper, save_path, size=4e5):
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        vocab_list = helper.id2tok
        glove_path = os.path.join(glove_dir, "glove.6B.{}d.txt".format(glove_dim))
        embed_save_path = os.path.join(save_path, "embedding_{}.npz".format(glove_dim))
        com_dict_save_path = os.path.join(save_path, "common_words_dict.npz")
        voc_embed = np.random.randn(len(vocab_list),glove_dim)
        com_embed= []
        com_dict={}
        found = 0
        with open(glove_path, 'r',encoding='utf8') as file:
                i= 0
                for line in tqdm(file, total=size):
                        array = line.lstrip().rstrip().split(" ")
                        word = array[0]
                        vector = list(map(float, array[1:]))
                        if word in vocab_list:
                                idx = vocab_list.index(word)
                                voc_embed[idx, :] = vector
                                found += 1
                        else:
                                if i < 15000:
                                        com_embed.append(vector)
                                        com_dict[word] = i
                                        i+=1                                      
        print("{}/{} of word in the vocabulary have corresponding vectors in glove".format(found, len(vocab_list)))
        np.savez_compressed(embed_save_path, vocab=voc_embed,common=com_embed)
        np.savez_compressed(com_dict_save_path,dictionary = com_dict)



def get_batches(inputs,labels, seq_len):
    assert inputs.shape == labels.shape
    num_batches = inputs.shape[1]//seq_len
    for batch_indx in tqdm(range(num_batches)):
        start_indx = batch_indx*seq_len
        end_indx = start_indx+seq_len
        inputs_batch = inputs[:,start_indx:end_indx]
        labels_batch = labels[:,start_indx:end_indx]
        yield inputs_batch,labels_batch


if __name__ =='__main__':
        country_code = 'FRA'
        helper, data = load_and_preprocess_data('input/un-general-debates.csv','script/data',country_code)
        embed_path = 'script/embed/{}'.format(country_code)
        if not os.path.exists(embed_path):
                process_glove(200, 'input/glove.6B', helper,embed_path)
        helper.add_common_words_dict(embed_path)
	
