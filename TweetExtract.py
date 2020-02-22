import pandas as pd
#   GetOldTweets3 --username "realDonaldTrump" --since 2017-1-20 --until 2020-01-14
df = pd.read_csv('Tweet_Data/Tweets_20012017_14012020.csv')

date = []
time  = []

for index, row in df.iterrows():
    date.append(row.date[:10])
    time.append(row.date[11:])
data = pd.DataFrame({'Date': pd.Series(date), 'Time':pd.Series(time), 'Tweet': df.text})
