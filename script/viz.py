from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

## Filter for stop words    
def frequent_non_stop_word(count,number): 
    stop_words = stopwords.words("english")
    word_return = []
    freq_return = []
    i = 1
    while len(word_return)<number:
        if count[i][0] not in stop_words:
            word_return.append(count[i][0])
            freq_return.append(count[i][1])
        i+=1
    return word_return, freq_return

#### Visualization    
def loss(loss, filename,epoch = None):
    plt.figure(figsize=(18, 18))
    if epoch:
        plt.plot(epoch,loss)
    else:
        plt.plot(loss)
    plt.savefig(filename)

def embedding(tokens,embedding,wordlist,wordsize,filename,title):
    wordlist_indx = [tokens[w] for w in wordlist]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embedding[wordlist_indx, :])
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(wordlist):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y , color=['yellow'], s=wordsize[i])
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.title(title)
    plt.savefig(filename)
