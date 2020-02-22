import pandas as pd
from SubtaskOne_SignificantWords.Probability_of_discriminative_words import preprocess
from SubtaskOne_SignificantWords.key_words_naive_bayes_old import _tokenize_sentence
import preprocessor as p
from keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('/Users/tom/PycharmProjects/MastersThesis/Tweet_Data/vol_tweets.csv')
print(df.head(5))
vocab_list = preprocess(df)
print(len(vocab_list))
print(vocab_list)


def rewrite_tweets(df_high, vocab_list):
    clean_tweets = []
    embedded_tweets =[]
    exclude= ['http', '#', '@', 'tweet', 'pic']
    max_length = 0
    for tweet in df_high.text:
        tweet = _tokenize_sentence(p.clean(tweet), 1)
        tweet_tokens = []
        embedding = []
        for t in tweet:
            tweet_tokens.append(((((((((((t.replace('"', '')).replace('...', '')).replace('“', "")).replace('/', '')).replace(':', '')).replace('(', '')).replace(')', '')).replace('!', '')).replace('”', '')).replace("‘", "")).replace("’", ""))
            if tweet_tokens[-1]=='':
                tweet_tokens.pop(-1)
            else:
                embedding.append(vocab_list.index(tweet_tokens[-1]))
        embedded_tweets.append(embedding)
        clean_tweets.append(tweet_tokens)
        if len(embedding)>max_length:
            max_length =len(embedding)
    return clean_tweets, embedded_tweets, max_length



df['clean'], df['embedded'], max_length = pd.Series(rewrite_tweets(df, vocab_list))
print(df.head(4))
print(df.columns)
#df['embedded'] = pad_sequences(df['embedded'], maxlen=max_length)
#print(df['embedded'])

mean, std = df['volatility'].mean(), df['volatility'].std()
classes = []
for index, row in df.iterrows():
    if row.volatility > mean +2*std:
        classes.append('High_vol')
    elif mean+2*std >= row.volatility > mean+std:
        classes.append('Med_High_vol')
    elif mean+std> row.volatility >= mean:
        classes.append('Slight_High_vol')
    elif mean > row.volatility >= mean-std:
        classes.append('Sligh_Low_vol')
    elif mean-std>row.volatility >=mean-2*std:
        classes.append('Med_Low_vol')
    elif mean-2*std > row.volatility:
        classes.append('Low_vol')
df['Classes'] = pd.Series(classes)
print(df['Classes'].value_counts())
