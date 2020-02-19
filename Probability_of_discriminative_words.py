import pandas as pd
from key_words_naive_bayes import sig_words, _tokenize_sentence, low_vol, high_vol
from sklearn.feature_extraction.text import CountVectorizer
from many_hot_encoddings import vocab_creator
import numpy as np
import preprocessor as p

high_vol=high_vol.reset_index().drop(columns=['index'])
low_vol=low_vol.reset_index().drop(columns=['index'])


def preprocess(df_high, df_low):
    vocab= []
    exclude= ['http', '#', '@', 'tweet', 'pic']
    for tweet in df_high.text:
        tweet = _tokenize_sentence(p.clean(tweet), 1)
        for t in tweet:
            vocab.append(((((((((((t.replace('"', '')).replace('...', '')).replace('“', "")).replace('/', '')).replace(':', '')).replace('(', '')).replace(')', '')).replace('!', '')).replace('”', '')).replace("‘", "")).replace("’", ""))
    for tweet in df_low.text:
        tweet = _tokenize_sentence(p.clean(tweet), 1)
        for t in tweet:
            vocab.append(((((((((((t.replace('"', '')).replace('...', '')).replace('“', "")).replace('/', '')).replace(':', '')).replace('(', '')).replace(')', '')).replace('!', '')).replace('”', '')).replace("‘", "")).replace("’", ""))
    return list(set(vocab))




def get_probs(df_high, df_low):
    new_df = []
    for tweet in df_high.text:
        tweet = _tokenize_sentence(p.clean(tweet), 1)
        new_tweet = []
        for t in tweet:
            new_tweet.append(((((((((((t.replace('"', '')).replace('...', '')).replace('“', "")).replace('/', '')).replace(':', '')).replace('(', '')).replace(')', '')).replace('!', '')).replace('”', '')).replace("‘", "")).replace("’", ""))
        new_df.append(" ".join(new_tweet))
    for tweet in df_low.text:
        tweet = _tokenize_sentence(p.clean(tweet), 1)
        new_tweet = []
        for t in tweet:
            new_tweet.append(((((((((((t.replace('"', '')).replace('...', '')).replace('“', "")).replace('/', '')).replace(':', '')).replace('(', '')).replace(')', '')).replace('!', '')).replace('”', '')).replace("‘", "")).replace("’", ""))
        new_df.append(" ".join(new_tweet))
    vec = CountVectorizer()
    X = vec.fit_transform(new_df)
    return X.toarray(), vec.get_feature_names()

labels = ['High']*len(high_vol)+['Low']*len(low_vol)
vocab  =preprocess(high_vol, low_vol)

df = pd.DataFrame({'vocab':vocab})
counted_vectors,feature_names = get_probs(high_vol,low_vol)

vectors_high, vectors_low = counted_vectors[:int(len(counted_vectors)/2)], counted_vectors[int(len(counted_vectors)/2):]
print(len(vectors_high))
print(len(vectors_low))
x=1

