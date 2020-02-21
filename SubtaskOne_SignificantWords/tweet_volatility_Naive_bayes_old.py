import pandas as pd
from SubtaskOne_SignificantWords.key_words_naive_bayes_old import sig_words, _tokenize_sentence, low_vol, high_vol
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

'''
Useful Functions:
    populate_df_keywords
    
'''
high_words, low_words, both = sig_words()
key_words = high_words+low_words+both


high_vol=high_vol.reset_index().drop(columns=['index'])
low_vol=low_vol.reset_index().drop(columns=['index'])
high_vol=pd.concat([high_vol, pd.Series(['HIGH']*len(high_vol))], axis =1, ignore_index=True)
high_vol.columns = ['datetime', 'volatility', 'text', 'label']
low_vol = pd.concat([low_vol, pd.Series(['LOW']*len(low_vol))], axis=1, ignore_index=True)
low_vol.columns = ['datetime', 'volatility', 'text', 'label']


def populate_df_keywords(empty, key_words, vol_df):
    for index, row in vol_df.iterrows():
        tokens = _tokenize_sentence(row.text, 1)
        #print(tokens)
        for key in key_words:
            #print(index,key)
            if key in tokens:
                empty['{}'.format(key)][index] =1
            else:
                empty['{}'.format(key)][index]=0
    return empty

df_high = populate_df_keywords(pd.DataFrame(index = range(len(high_vol)), columns=key_words), key_words, high_vol)
df_high['LABELS'] = high_vol['label']
df_low = populate_df_keywords(pd.DataFrame(index = range(len(low_vol)), columns=key_words), key_words, low_vol)
df_low['LABELS']=low_vol['label']
df = pd.concat([df_low, df_high], axis=0).reset_index().drop(columns = ['index'])
labels = df.pop('LABELS')


nb = GaussianNB()
x_train, x_test, y_train, y_test = train_test_split(df, labels, shuffle =True, test_size=0.33)
fitted = nb.fit(x_train, y_train)
train_pred = nb.predict(x_train)
test_pred = nb.predict(x_test)
train_report = classification_report(y_train, train_pred)
test_report = classification_report(y_test, test_pred)



