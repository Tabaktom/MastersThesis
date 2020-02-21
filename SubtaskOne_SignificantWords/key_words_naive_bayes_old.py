import pandas as pd
from nltk.tokenize import word_tokenize
import string
from collections import Counter
'''
Useful Functions:
    glue_tokens
    unglue_tokens
    _tokenize_sentence
    unigram
    word_dist
'''

def glue_tokens(tokens, order):
    return '{0}@{1}'.format(order, ' '.join(tokens))

def unglue_tokens(tokenstring, order):
    if order == 1:
        return [tokenstring.split("@")[1].replace(" ","")]
    return tokenstring.split("@")[1].split(" ")

def _tokenize_sentence(sentence, order):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = ((sentence.replace('Ô¨Å', '*').replace('Ô¨Ç', '*')).replace('Äö', "'")).replace('Ô¨Ç', '*')
    sentence = ((((sentence.strip(string.punctuation)).replace(",", "")).replace("'", "")).replace('.', '')).replace('...', '')
    sentence = sentence.replace('?', '')
    tokens = sentence.split()
    tokens = ['<s>'] * (order - 1) + tokens + ['</s>']
    return tokens

def unigram(corpus, label, target):
    unigrams = Counter()
    for sent, lab in zip(corpus, label):
        if isinstance(sent, str) == True:
            if lab == target:
                words = _tokenize_sentence(sent, 1)
                # print(words)
                # print("tokenized", words)
                for w in words:
                    unigrams[w] += 1
    unigram_total = sum(unigrams.values())
    frequent_vocab = []
    for w, v in unigrams.items():
        if v >= 2:
            frequent_vocab.append(w)

    context = []
    uni = Counter()
    for sent, lab in zip(corpus, label):
        if isinstance(sent, str) == True:
            if lab == target:
                words = tokenize_sentence(sent, 1)
                for w in words:
                    if w not in frequent_vocab:
                        word = '<unk/>'
                    else:
                        word = w
                    context.append(word)
                    uni[word] += 1
    total = sum(uni.values())
    return uni, total


df=pd.read_csv(r'../Tweet_Data/vol_tweets.csv')
df =df.sort_values(['volatility'])

tenpercentile = int(len(df)*0.2)
ninetypercentile=int(len(df)*0.8)
low_vol = df[:tenpercentile+1]
high_vol=df[ninetypercentile:]
low_D = Counter()
high_D = Counter()

def word_dist(sent, D):
    tok_sent = _tokenize_sentence(sent, 1)
    for word in tok_sent:
        if word not in D:
            D[word] =1
        else:
            D[word]+=1
    return D

for index, row in low_vol.iterrows():
    low_D = word_dist(row.text, low_D)
    low_values = [val[1] for val in low_D.items()]
    low_words = [val[0] for val in low_D.items()]
for index, row in high_vol.iterrows():
    high_D = word_dist(row.text, high_D)
    high_words = [val[0]for val in high_D.items()]
    high_values = [val[1]for val in high_D.items()]


high_df = pd.DataFrame({'word': pd.Series(high_words), 'values':pd.Series(high_values)})
low_df = pd.DataFrame({'word': pd.Series(low_words), 'values':pd.Series(low_values)})
high_df = high_df.sort_values('values', ascending=False).where(high_df['values']>25).where(high_df['values']<51).dropna().reset_index().drop(columns=['index'])
low_df = low_df.sort_values('values', ascending=False).where(low_df['values']>25).where(low_df['values']<51).dropna().reset_index().drop(columns=['index'])


def sig_words():
    ###5 Percentile key words ###
    #high = ['democrats', 'news', 'country', 'witch', 'job', 'hunt', 'fake', 'russia', 'korea', 'tax', 'trade', 'obama']
    #low = ['american', 'military', 'vote', 'congress', '@foxnews']
    #both = ['president', 'republican', 'trump', 'border']

    ### 10 Percentile key words ###
    #high = ['news', 'crime', 'russia', 'obama', 'korea', 'security', 'campaign', 'witch', 'fbi', 'imigration', 'tax', 'stock', 'market']
    #both = ['president', 'democrats', 'country', 'fake', 'vote', 'military', 'trump', 'congress', 'border', 'trade', 'media']
    #low = ['america', 'republican', 'senate', 'governor', 'impeachment', '@foxnews', 'china']

    ## 15 Percentile key words ###
    #high = ['democrats', 'president', 'trump', 'crime', 'russia', 'military', 'dems', 'job', 'media', 'korea', 'obama', 'china', 'fbi', 'campaign', 'witch', 'hunt']
    #low = ['republican', 'honor', 'trade', 'congratulations', 'tax']
    #both =['country', 'fake', 'news', 'jobs', 'vote', 'american', 'america']

    ## 20 Percentile key words ##
    high = ['country', 'border', 'crime', 'dems', 'media', 'russia', 'korea', 'hunt', 'witch', 'obama', 'security', '@foxandfriends', 'congress', 'immigration', 'fbi', 'china']
    low =['america',  'vote', 'congratulations']
    both = ['president', 'news', 'fake', 'democrats', 'trade', 'jobs', 'republicans', 'trump', 'american','military', 'tax']
    return high, low, both




