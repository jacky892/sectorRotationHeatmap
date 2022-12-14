import pandas as pd
import base64
import os,sys,gzip

def download_cmc_crypt_data(ticker='BTC', cmcpath='data'):
    import os
    from cryptocmd import CmcScraper
    # initialise scraper with time interval for e.g a year from today

    # Pandas dataFrame for the same data
    if not ticker[-2:]=='=C':
        ticker=f'{ticker}=C'
    scraper = CmcScraper(ticker[:-2], "12-06-2021")
    df = scraper.get_dataframe()
    if not os.path.exists(f'{cmcpath}'):
        os.makedirs(cmcpath)
    if not os.path.exists(f'{cmcpath}/{ticker}'):
        os.makedirs(f'{cmcpath}/{ticker}')
    ofname=f'{cmcpath}/{ticker}/{ticker}.1d.csv.gz'

    df.to_csv(ofname, compression='gzip')
    print(f'data saved to {ofname}')
    return ofname

def read_cmc_crypto_quote(ticker='BTC', cmcpath='data', pred_date=None):
    import pandas as pd
    import numpy as np
    if not ticker[-2:]=='=C':
        ticker=f'{ticker}=C'

    fname=f'{cmcpath}/{ticker}/{ticker}.1d.csv.gz'
    df=pd.read_csv(fname, compression='gzip', parse_dates=True, index_col=0)
    df.sort_values(by='Date', inplace=True)
    try:
        df['Date']=df.Date.apply(lambda x:np.datetime64(x))
    except Exception as e:
        print('exception ',e) 
        print(df.tail())
    df.set_index('Date', inplace=True)
    df['ncoins']=df['Market Cap']/df['Close']
    if pred_date is None:
        return df
    return df.loc[:pred_date]

def is_console_mode():
    import sys
    if sys.stdin and sys.stdin.isatty():
        return True
    return False

def is_fx_ticker(ticker):
    if ticker==ticker.lower():
        return True
    return False

def is_cmc_crypto(ticker):
    if ticker[-2:]=="=C":
        return True
    return False


def read_duka_dl(ticker='cadjpy'):   
    import pandas as pd
    today=pd.to_datetime('today').strftime('%Y-%m-%d')  
    today_28ago=pd.to_datetime('today')-pd.to_timedelta('80  days')
    today_28ago_str=today_28ago.strftime('%Y-%m-%d')
    cmd=f'npx dukascopy-node -i {ticker} -from {today_28ago_str} -to {today} -t h4 -f csv'
    import sys 
    import os
    ofname=f'download/{ticker}-h4-bid-{today_28ago_str}-{today}.csv'
    if not os.path.exists(ofname):
        print(f'no previous download, now download data {ofname}')
        print(cmd)                    
       
        os.system(cmd)
    return ofname


def pickle_output_dict_with_img(obj_dict, pklfname='ref_userid.pkl.gz', get_b64=True):
    '''
    it will read all img file name list in 'img_list' params and 
    load them using PIL.Image
    '''
    import pickle, gzip,os
    from PIL import Image    
    save_dict=obj_dict.copy()
    if 'img_list' in save_dict:
        img_list=save_dict['img_list']
    else:
        print('missed img:' , save_dict.keys())
        img_list=[]
    save_dict['img_obj']=[]

    for imgfname in img_list:      
        print('reading image ',imgfname)
        if '.jpg' in imgfname or '.png' in imgfname:
            save_dict['img_obj'].append(Image.open(imgfname))
    print(save_dict)
    dirname=os.path.dirname(pklfname)
    if not os.path.exists(dirname) and len(dirname)>1:
        os.makedirs(dirname)
    with gzip.open(pklfname, 'wb') as f:
        print('f handle:',f)
        pickle.dump(save_dict,f, protocol=pickle.HIGHEST_PROTOCOL)
    if get_b64:
        with gzip.open(pklfname, 'rb') as f:
            b64_objmsg = base64.b64encode(f.read())
        return b64_objmsg


def load_output_dict_pickle_with_img(pklfname='ref_userid.pkl.gz'):
    import pickle, gzip
    from PIL import Image
    load_dict=None
    with gzip.open(pklfname, 'rb') as f:
        load_dict=pickle.load(f)
    if load_dict is None:
        return None
    
    if 'img_obj' in load_dict:
        print(load_dict)
        img_obj=load_dict['img_obj']
        for img in img_obj:
            if not is_console_mode():
                img.show()
            
    if 'msg' in load_dict:
        msg=load_dict['msg']
        
    if 'send_df' in load_dict:
        send_df = load_dict['send_df']
    return load_dict


def load_dukas_df(fname):
    import pandas as pd
    import random
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
    df2['Volume']=10
    df2['Volume']=df2.Volume.apply(lambda x:x+random.randint(0,5))
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

def trim_img(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def show_image(fname, show=True, trim=False):

    def show_img(fname, show, trim):
        from IPython.display import Image
        Image(filename=fname)
        from PIL import Image, ImageChops
        im = Image.open(fname)
        if trim:
            im = trim_img(im)
            im.save(fname)
        if show or not is_console_mode():
            print("asdjflasjdflaksd showing image alsdjflasdjf", is_console_mode())
            im.show()

    show_img(fname, show, trim)
    return fname

class commonUtil:
    @staticmethod
    def download_cmc_crypt_data(ticker, cmcpath='data'):
        return download_cmc_crypt_data(ticker, cmcpath)
    @staticmethod
    def pickle_output_dict_with_img(obj_dict, pklfname='ref_userid.pkl.gz'):
        return pickle_output_dict_with_img(obj_dict, pklfname=pklfname)

    @staticmethod
    def load_output_dict_pickle_with_img(pklfname='ref_userid.pkl.gz'):
        return load_output_dict_pickle_with_img(pklfname)

    @staticmethod
    def read_quote(ticker, pred_date=None):
        if is_fx_ticker(ticker):
            return read_fx_quote(ticker, pred_date=pred_date)
        elif is_cmc_crypto(ticker):
            return read_cmc_crypto_quote(ticker, pred_date=pred_date)
        else:
            return read_yf_quote(ticker, pred_date=pred_date)

    @staticmethod
    def download_quote(ticker):
        if ticker.lower()==ticker:
            ret=read_duka_dl(ticker)
        elif is_cmc_crypto(ticker):
            ret=download_cmc_crypt_data(ticker)
        else:
            ret=commonUtil.download_yf_quote(ticker)
        return ret

    @staticmethod
    def getProp(key):
        key_dict={}
        key_dict['dataroot']='/Users/jackylee/optiondata'
        key_dict['enable_static_cache']=False
    #    key_dict['dataroot']='local_dataroot/optiondata'
        key_dict['gwip']='192.168.8.101'
        key_dict['rqip']='192.168.8.112'
        key_dict['rqip']='localhost'
        #key_dict['rbmq_ip']='localhost'
        key_dict['rbmq_ip']='127.0.0.1'
        key_dict['nfsip']='192.168.8.112'
        key_dict['rq_username']='tgbot'
        key_dict['rq_password']='tgbot'
        if key is None:
            return key_dict
        if not key in key_dict.keys():
            return None
        return  key_dict[key]

    @staticmethod
    def show_image(fname, show=True, trim=False):
        return show_image(fname, show, trim)

    @staticmethod
    def is_console_mode():
        return is_console_mode()

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
