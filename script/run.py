import argparse
import os.path
import rnn
import skipgram
import util


	
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("country_code",help="Enter the ISO-3166 code of the target country")
        parser.add_argument("operation",help="Train a new model or generate text with existing model",choices=['train','generate'])
        args = parser.parse_args()
        embed_path = 'script/embed/{}'.format(args.country_code)  
        helper, data_raw = util.load_and_preprocess_data('input/un-general-debates.csv','script/data',args.country_code)
        if not os.path.exists(embed_path):
                util.process_glove(200, 'input/glove.6B', helper, embed_path)
        helper.add_common_words_dict(embed_path)
        if args.operation =='generate':
                while True:
                        word_in_vocab = True
                        prime = input("Please enter a prime word to start text generation: ").strip().lower()
                        prime_id = helper.tok2id.get(prime,None)
                        if not prime_id:
                                prime_id = helper.cwtok2id.get(prime,None)
                                word_in_vocab = False
                        if prime_id:
                                break
                rnn.generate_w_random_search(args.country_code,embed_path, helper, prime_id, prime,word_in_vocab,10,5)
        else:
                epochs = input("Please enter the number of epochs to train (default to 50) : ")
                try:
                        epochs = int(epochs)
                except:
                        epochs = 50
                print('Start training for {} epochs. This is going to take a while...'.format(epochs))
                rnn.do_train(args.country_code,embed_path,helper,data_raw,epochs,True)

