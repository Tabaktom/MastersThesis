import pandas as pd
from SubtaskOne_SignificantWords.key_words_naive_bayes_old import _tokenize_sentence, low_vol, high_vol
from sklearn.feature_extraction.text import CountVectorizer
high_vol=high_vol.reset_index().drop(columns=['index'])
from sklearn.naive_bayes import GaussianNB, MultinomialNB

'''
Useful Functions:
    vocab_creator
    create_corpus
'''

low_vol=low_vol.reset_index().drop(columns=['index'])
labels = ['High']*len(high_vol) +['Low']*len(low_vol)

def vocab_creator(df_high, df_low):
    vocab_high = []
    vocab_low =[]
    for index, row in df_high.iterrows():
        tokens = _tokenize_sentence(row.text, 1)
        for t in tokens:
            if t not in vocab_high:
                vocab_high.append(t)
    for index, row in df_high.iterrows():
        tokens = _tokenize_sentence(row.text, 1)
        for t in tokens:
            if t not in vocab_low:
                vocab_low.append(t)
    vocab = set(vocab_high +vocab_low)
    return list(vocab)


def create_corpus(df_high, df_low):
    corpus =[]
    for index, row in df_high.iterrows():
        corpus.append(row.text)
    for index, row in df_high.iterrows():
        corpus.append(row.text)
    vec = CountVectorizer()
    X = vec.fit_transform(corpus)
    return X.toarray(), vec.get_feature_names()

vectors, features_names = create_corpus(high_vol, low_vol)

nb = GaussianNB()
MNB = MultinomialNB()
fitted = MNB.fit(vectors, labels)
params = fitted.feature_count_

Meaning = pd.DataFrame({'word': features_names, 'weight':params[0]})
Meaning.index= Meaning['word']
Meaning=Meaning.drop(columns= ['word'])
Meaning = Meaning.sort_values(by = 'weight', ascending=False)
