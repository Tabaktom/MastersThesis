import pandas as pd
import numpy as np
import plotly.graph_objects as go

'''
Useful Functions:
    plot.pdf
    plot.cdf
'''
df = pd.read_csv('Financial_Data/GBPUSD_1min_local_full_new.txt')

df['Datetime'] = df['Date'] + pd.Series([' ']* len(df)) + df['Time']
df['Datetime'] =pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
df=df.set_index('Datetime')
df = df.drop(columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Up', 'Down'])
df['pct_change']=df.Close.pct_change()
df['log_ret']=np.log(df.Close)-np.log(df.Close.shift(1))
#df['volatility']=df.log_ret.expanding(5).std()
rev = df[::-1]
rev['volatility'] = rev.log_ret.rolling(5).std()
df=rev[::-1]
mean, std = df.volatility.mean(), df.volatility.std()
class plot():
    def pdf(mean, std, df, tmean, tstd, tdf, nmean, nstd, ndf):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x = df['volatility'],
                                   histnorm='probability',
                                   name='Normal Volatility',
                                   opacity=1,
                                   xbins=dict(
                                       start=mean-std*100,
                                       end=mean+std*10,
                                       size=std/8),
                                   marker=dict(color='yellow')))

        fig.add_trace(go.Histogram(x = tdf['volatility'],
                                   histnorm='probability',
                                   name='Tweets',
                                   opacity=0.65,
                                   xbins=dict(
                                       start=tmean-tstd*100,
                                       end=tmean+tstd*10,
                                       size=tstd/8),
                                   marker=dict(color='red')))

        fig.add_trace(go.Histogram(x = ndf['volatility'],
                                   histnorm='probability',
                                   name='Non-Tweets',
                                   opacity=0.35,
                                   xbins=dict(
                                       start=nmean-nstd*100,
                                       end=nmean+nstd*10,
                                       size=nstd/8),
                                   marker=dict(color='blue')))
        fig.add_shape(go.layout.Shape(type ='line', x0=mean, y0=0, x1=mean, y1=0.1,
                                      name='Volatility Mean',line=(dict(color='blue'))))
        fig.add_shape(go.layout.Shape(type ='line', x0=tmean, y0=0, x1=tmean, y1=0.1,
                                      name='Twitter Volatility Mean',line=(dict(color='red'))))
        fig.add_shape(go.layout.Shape(type ='line', x0=nmean, y0=0, x1=nmean, y1=0.1,
                                      name='Non-Twitter Volatility Mean',line=(dict(color='yellow'))))
        fig.update_layout(barmode='overlay', title='Probability Distribution Function', xaxis_title='Volatility', yaxis_title='Probability')
        fig.show()

    def cdf(df_vol, dft_vol, dfn_vol):
        fig = go.Figure(data=[go.Histogram(x=df_vol, cumulative_enabled=True, opacity=0.5,
                                           histnorm='probability', name='Entire dataset Cdf',
                                           marker=(dict(color='green')))])
        fig.add_trace(go.Histogram(x=dfn_vol, cumulative_enabled=True, opacity=0.5, histnorm='probability',
                                   name='Non-Tweet Cdf', marker=(dict(color='blue'))))
        fig.add_trace(go.Histogram(x=dft_vol, cumulative_enabled=True, opacity=0.5, histnorm='probability',
                                   name='Tweet Cdf', marker=(dict(color='red'))))
        fig.update_layout(title='Cumulative Distribution Frequency', xaxis_title='Volatility', yaxis_title='Cumulative Probability')
        fig.show()

tweetdf = pd.read_csv('Tweet_Data/Tweets_20012017_14012020.csv')

tweetdf = tweetdf.drop(columns = ['username', 'to', 'replies', 'retweets', 'favorites',
                                  'geo', 'mentions', 'hashtags', 'id', 'permalink'])
date_array = []
for ind, row in tweetdf.iterrows():
    date_array.append(row.date[:16])
tweetdf['Datetime'] = pd.to_datetime(pd.Series(date_array), format='%Y-%m-%d %H:%M')
tweetdf=tweetdf.groupby(['Datetime']).sum()
tweetdf = tweetdf.pop('text')

df_vol_tweets = pd.merge(df, tweetdf, on='Datetime')
df_vol_tweets = df_vol_tweets.drop(columns=['Close', 'pct_change', 'log_ret'])

df_non_tweet = df.drop(df_vol_tweets.index)
tweet_mean, tweet_std = df_vol_tweets['volatility'].mean(), df_vol_tweets['volatility'].std()
non_mean, non_std = df_non_tweet['volatility'].mean(), df_non_tweet['volatility'].std()
df_vol_tweets.to_csv('vol_tweets.csv')
#plot.pdf(mean, std, df, tweet_mean, tweet_std, df_vol_tweets, non_mean, non_std, df_non_tweet)
#plot.cdf(df['volatility'],df_vol_tweets['volatility'], df_non_tweet['volatility'])