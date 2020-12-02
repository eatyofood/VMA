import pandas as pd
import numpy as np
#import cufflinks as cf
#cf.go_offline(connected=False)
'''
pd.set_option('display.max_column',None)
df = pd.read_csv('data/BATS BEEM, 15.csv',index_col='time')
df.index = pd.to_datetime(df.index,unit='s')
df

# Functions



df = df[['open','low','high','close','Volume','VMA.6']]

df'''

def scale_to_close(df,col):
    col_name = col+'_scaled'
    df[col_name] = df[col].replace(True,1).replace(1,df.close)#.........
    return df,col_name

def VMA( input_list, indicator_precision=2, multiplier_var=1,prev_vidya=None, period=5):
    '''
    returns:
        a list: 
            VMA - Variable Moving Average ( also called vidya)
    '''
    vidya_list = []
    if prev_vidya is None:
        prev_vidya = 0.0
    multiplier = multiplier_var/(period+1)
    deltas = np.diff(input_list)
    #print(deltas)
    for index in range(period-1, len(deltas)):
        diff = deltas[index-period+1:index+1]
        up_sum = round(diff[diff >= 0].sum(),indicator_precision)
        down_sum = round(abs(diff[diff < 0].sum()),indicator_precision)
        if up_sum == 0 and down_sum == 0:
            cmo = 0.0
        else:
            cmo = abs((up_sum-down_sum)/(up_sum+down_sum))
        prev_vidya = round(((input_list[index+1] * multiplier * cmo) + (prev_vidya * (1-(multiplier*cmo)))), indicator_precision)
        vidya_list.append(prev_vidya)
    for i in range(period):
        vidya_list.insert(0,0)
    return vidya_list

def ST(df,f=3,n=7): #df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
    #Calculation of SuperTrend
    '''
    makes a strawberrie supertrend on your dataframe
    df= dataframe
    n = periods
    f = factor
    
    '''
    import pandas_ta as pta

    df['ATR'] = pta.atr(df.high,df.low,df.close)
    df['Upper Basic']=(df['high']+df['low'])/2+(f*df['ATR'])
    df['lower Basic']=(df['high']+df['low'])/2-(f*df['ATR'])
    df['Upper Band']=df['Upper Basic']
    df['lower Band']=df['lower Basic']
    for i in range(n,len(df)):
        if df['close'][i-1]<=df['Upper Band'][i-1]:
            df['Upper Band'][i]=min(df['Upper Basic'][i],df['Upper Band'][i-1])
        else:
            df['Upper Band'][i]=df['Upper Basic'][i]    
    for i in range(n,len(df)):
        if df['close'][i-1]>=df['lower Band'][i-1]:
            df['lower Band'][i]=max(df['lower Basic'][i],df['lower Band'][i-1])
        else:
            df['lower Band'][i]=df['lower Basic'][i]   
    df['SuperTrend']=np.nan
    for i in df['SuperTrend']:
        if df['close'][n-1]<=df['Upper Band'][n-1]:
            df['SuperTrend'][n-1]=df['Upper Band'][n-1]
        elif df['close'][n-1]>df['Upper Band'][i]:
            df['SuperTrend'][n-1]=df['lower Band'][n-1]
    for i in range(n,len(df)):
        if df['SuperTrend'][i-1]==df['Upper Band'][i-1] and df['close'][i]<=df['Upper Band'][i]:
            df['SuperTrend'][i]=df['Upper Band'][i]
        elif  df['SuperTrend'][i-1]==df['Upper Band'][i-1] and df['close'][i]>=df['Upper Band'][i]:
            df['SuperTrend'][i]=df['lower Band'][i]
        elif df['SuperTrend'][i-1]==df['lower Band'][i-1] and df['close'][i]>=df['lower Band'][i]:
            df['SuperTrend'][i]=df['lower Band'][i]
        elif df['SuperTrend'][i-1]==df['lower Band'][i-1] and df['close'][i]<=df['lower Band'][i]:
            df['SuperTrend'][i]=df['Upper Band'][i]
    df['bullish_supertrend'] = df['SuperTrend']<df['close']
    
    # scale the bool to lower band
    
    df['super_bull'] = df['bullish_supertrend'].replace(True,1).replace(1,df.super_lower)
    return df[['bullish_supertrend','ATR','super_bull']]


# i just mixed all the above functions in this one. 

def super_trend(df,factor = 3):
    '''
    takes a dataframe and 
    returns : 
        dataframe with:
            vma         - variable moving avg
            super_upper - ( band)
            super_lower - ( band)
            bullish_supertrend - bool
            super_bull_scale   - bullish_supertrend scaled to
                                 lower_band

    '''
    
    import pandas_ta as pta

    df['ATR'] = pta.atr(df.high,df.low,df.close)

    # create the vma
    df['vma'] = VMA(df.close,indicator_precision=2)


    # ADD THE BANDS
    
    df['super_lower']  = df['vma'] - (df['ATR']*factor)
    df['super_upper']  = df['vma'] + (df['ATR']*factor)
    df[['bullish_supertrend','ATR','super_bull_scale']] = ST(df)
    return df #df[['vma','super_lower','super_upper','bullish_supertrend','super_bull']]

## super_bull tracs the lower band when its bullish

