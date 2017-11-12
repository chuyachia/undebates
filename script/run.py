import tensorflow as tf
import util
import skipgram
import rnn

def main(country_code,prime_word,train= False):
    helper, data_raw = util.load_and_preprocess_data(data_path='../input/un-general-debates.csv',country_code=country_code)
    embedding = util.load_embeddings(country_code)
    if embedding is None:
        embedding = skipgram.do_train(country_code,helper,data_raw)
    if train:
        rnn.do_train(country_code,embedding,helper,data_raw)
    rnn.generate_w_random_search(country_code,embedding, helper, prime_word,10,5)
	
	
if __name__ == "__main__":
    main('FRA','terrorism')
