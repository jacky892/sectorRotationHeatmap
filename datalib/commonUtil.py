import pandas as pd
import os,sys,gzip

def is_fx_ticker(ticker):
    if ticker==ticker.lower():
        return True
    return False

def read_duka_dl(ticker='cadjpy'):   
    import pandas as pd
    today=pd.to_datetime('today').strftime('%Y-%m-%d')  
    today_14ago=pd.to_datetime('today')-pd.to_timedelta('28 days')
    today_14ago_str=today_14ago.strftime('%Y-%m-%d')
    cmd=f'npx dukascopy-node -i {ticker} -from {today_14ago_str} -to {today} -t h4 -f csv'
    import sys 
    import os
    ofname=f'download/{ticker}-h4-bid-{today_14ago_str}-{today}.csv'
    if not os.path.exists(ofname):
        print(f'no previous download, now download data {ofname}')
        print(cmd)                    
       
        os.system(cmd)
    return ofname



def load_dukas_df(fname):
    import pandas as pd
    df2=pd.read_csv(fname)                                                                                                                      
    import datetime               
    df2['Date']=df2.timestamp.apply(lambda x:datetime.datetime.fromtimestamp(x/1000))
    df2.set_index('Date', inplace=True)
    rename_dict={}
    rename_dict['open']='Open'
    rename_dict['high']='High'
    rename_dict['low']='Low'
    rename_dict['close']='Close'
    df2.rename(rename_dict, inplace=True, axis=1)

    return df2

def clean_old_fx_download():
    import glob
    import os
    qstr='download/*.csv'
    flist=glob.glob(qstr)
    for fname in flist:
        os.remove(fname)
        print('removing:',fname)

def read_fx_quote(ticker, pred_date=None):
    import os
    import glob
    fname=f'download/{ticker}-h4-bid-2022-11-11-2022-11-25.csv'

    if not os.path.exists(fname):
        flist=glob.glob(f'download/{ticker}*.csv')
        if len(flist)>0:
            fname=flist[0]
        else:
            fname=read_duka_dl(ticker)
    df=load_dukas_df(fname)
    if pred_date is None:
        return df
    return df.loc[:pred_date]


def update_fx_quote():
    fx_tlist=['eurusd', 'gbpusd', 'nzdusd', 'audusd', 'xauusd', 'usdjpy', 'usdchf', 'usdcad']
    for t in fx_tlist:
        #fname=read_duka_dl(t)
        df=read_fx_quote(t)

def read_yf_quote(ticker, pred_date=None):
    '''
    read quote from ./data/
    pred_date='20110901' for test
    '''
    dataroot='data'
    fname=f'{dataroot}/{ticker}/{ticker}.1d.csv.gz'
    df=None
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)
    if not os.path.exists(f'{dataroot}/{ticker}'):
        os.makedirs(f'{dataroot}/{ticker}')
    if os.path.exists(fname):
        df=pd.read_csv(fname, compression='gzip', index_col=0, parse_dates=True)
    dataroot='data'
    dirname=(f'{dataroot}/{ticker}')
    ofname=f'{dataroot}/{ticker}/{ticker}.1d.csv.gz'
    if not os.path.exists(f'{dataroot}/{ticker}'):
        os.makedirs(dirname)
    if df is None:
        df=pd.read_csv(ofname, compression='gzip', index_col=0, parse_dates=True)
    else:
        df.to_csv(ofname, compression='gzip')
#        
    if pred_date is None:
        return df
    return df.loc[:pred_date]


class commonUtil:

    @staticmethod
    def read_quote(ticker, pred_date=None):
        if is_fx_ticker(ticker):
            return read_fx_quote(ticker, pred_date=pred_date)
        else:
            return read_yf_quote(ticker, pred_date=pred_date)


    @staticmethod
    def download_yf_quote(ticker):
        import pandas as pd
        import yfinance as yf
        import os, gzip
        stk=yf.Ticker(ticker)
        csv_dir=f'data/{ticker}'
        csv_fname=f'{csv_dir}/{ticker}.1d.csv.gz'
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        df=None
        try:
            df=stk.history(period='max')
            df.to_csv(csv_fname, compression='gzip')
            print(f'data for {ticker} shape:%s, late:%s' % (df.shape,  df.index[-1]))
        except Exception as e:
            print('error downloading new quotes:',e)
            df=pd.read_csv(csv_fname, compression='gzip')
        return df
