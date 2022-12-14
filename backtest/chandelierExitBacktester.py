### 
#
'''
run backtesting according to the signed entry bars using chandelier exit (recent high - n * ATR)
required paramerters include threshold of N-bars movement (e.g. > 10% in 7 trading bars)
retrace_atr_multiple = 3
pdf/price_df being price quote from commonUtil.read_quote
def_pct_stop = default percentage stop loss from entry price
'''
import os
import logging
from collections import defaultdict
from datalib.commonUtil import commonUtil as cu
import pandas as pd

logging.basicConfig()
logger=logging.getLogger()
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.WARN)

def get_next_day(datestr, pdf):
    '''
    get the next trading after datestr from quote=pdf
    '''
    _=pdf.loc[datestr:]
    if len(_)==0:
        return pd.to_datetime('20000101')
    idx=0 if len(_)<=1 else 1
    return _.index[idx]

def dlog(v, *argv):
    '''call dlog for each input argument'''
    dlog1(v)
    for v1 in argv:
        dlog1(v1)

def dlog1(msg):
    '''
    if msg is  dataframe and in jupyter, use display, otherwise, use logger.debug
    '''
    if type(msg) in [type(pd.DataFrame(dtype='float64')), type(pd.Series(dtype='float64'))]:
        showtable(msg)
    else:
        print(msg)
        logger.debug(msg)

def showtable(df):
    '''show table if from jupyter, just print if not'''
    if isnotebook():
        display(df)
    else:
        print(df)

def isnotebook():
    ''' check if running from jupyter'''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def add_3way_label(ndf, uplabel_col='_lb_7_bigrise', dnlabel_col='_lb_7_bigdrop', newcol='_lbm_7_bigmove'):
    '''
    group two columns (dnlabel_col, uplabel_col) of up/down into 3-class label of 1, 0, -1  into a new columns: newcol
    '''
    y=ndf[uplabel_col].copy()
    bidx=ndf[dnlabel_col]>0
    idx=y[bidx].index
    y.loc[idx]=-1
    ndf[newcol]=y
    return ndf



def get_long_max_drawdown_details(pdf, day_cnt=250, atr_bars=22, atr_multiple=4, plot=False)->dict:
    '''
    get the max draw down from peak during the timeframe in pdf
    return dict
        dict keys includes ['mdd_er', 'ret_ser, 'mdd', 'atr_exit,  'pct_ret, pct_ret_if_exceeded'
             'atr_exit_dd', 'dd_if_exited', 'mdd_exeeded', 'mdd_idx']
        mdd is the single mdd value
    '''
    #import pandas_ta
#    Roll_Max = pdf.High.rolling(window=day_cnt, min_periods=1).max()
#    Daily_Drawdown =pdf.Low/Roll_Max - 1.0
    if type(pdf)==type(pd.Series(dtype='float64')):
        _df=pd.DataFrame()
        _df['High']=pdf
        _df['Open']=pdf
        _df['Close']=pdf
        _df['Low']=pdf
        pdf=_df
    Roll_Max = pdf.High.rolling(window=day_cnt, min_periods=1).max()
    Daily_Drawdown =pdf.Low/Roll_Max - 1.0

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window=250, min_periods=1).min()
    bidx=(Daily_Drawdown==Max_Daily_Drawdown)
    mdd_idx=Max_Daily_Drawdown[bidx].index

    ret_dict=defaultdict()
    ret_dict['mdd_ser']=Max_Daily_Drawdown
    ret_dict['ret_ser']=(pdf.Close/pdf.Close.iloc[0])-1
    ret_dict['mdd']= Max_Daily_Drawdown.min()
    if 'atr' in pdf.columns:
        atr=pdf['atr']
    else:
        atr=pdf.ta.atr(atr_bars)
    ret_dict['atr_exit']=Roll_Max-atr*atr_multiple
    ret_dict['atr_exit_dd']=ret_dict['atr_exit']/Roll_Max-1
    ret_dict['dd_if_exited']=(Roll_Max-pdf.Close)/Roll_Max
    ret_dict['dd_margin_to_exit']=ret_dict['atr_exit_dd']-ret_dict['mdd_ser']
    ret_dict['mdd_exceeded']=ret_dict['dd_margin_to_exit'].any()<0
    ret_dict['mdd_idx']=mdd_idx
    ret_dict['pct_ret']=(pdf.Close/pdf.Close.iloc[0])-1
    ret_dict['pct_ret_if_exited']=ret_dict['pct_ret'].loc[mdd_idx[-1]]

    if plot:
        plot_df=pd.DataFrame()
        plot_df['daily_drawdown']=Daily_Drawdown
        plot_df['equity']=ret_dict['ret_ser']
        plot_df['max_drawdown']=Max_Daily_Drawdown
        ax=plot_df.plot(figsize=(12,8))
        ax.scatter(x=mdd_idx, y=plot_df.loc[mdd_idx].max_drawdown, marker='^', s=180, color='r')
        for ts in mdd_idx:
            ax.axvline(ts)

    return ret_dict

def show_trades(trades_df):
    '''
    display most essential fields of trades_df
    '''
    dlog(trades_df.columns)
    cols=['ticker', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pct_profit', 'day_in_trade', 'annualized_profit']
    dlog(trades_df[cols])
    return trades_df[cols]



def get_short_max_drawdown_details(pdf, day_cnt=250, atr_bars=22, atr_multiple=4, plot=False)->dict:
    '''
    sell get_long_max_drawdown_details
    '''
    #import pandas_ta
    #if type(pdf)==type(pd.DataFrame()):
    if isinstance(pdf,pd.DataFrame):
        Roll_Min = pdf.Low.rolling(window=day_cnt, min_periods=1).min()
        Daily_Drawdown =-(pdf.High/Roll_Min - 1.0)
    else:
        Roll_Min.rolling(window=day_cnt, min_periods=1).min()
        Daily_Drawdown =-(pdf/Roll_Min - 1.0)
    #if type(pdf)==type(pd.Series()):
    if isinstance(pdf,pd.Series):
        _df=pd.DataFrame()
        _df['High']=pdf
        _df['Open']=pdf
        _df['Close']=pdf
        _df['Low']=pdf
        pdf=_df

    Roll_Min = pdf.Low.rolling(window=day_cnt, min_periods=1).min()
    Daily_Drawdown =1- pdf.High/Roll_Min

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window=250, min_periods=1).min()
    bidx=(Daily_Drawdown==Max_Daily_Drawdown)
    mdd_idx=Max_Daily_Drawdown[bidx].index

    if plot:
        Daily_Drawdown.plot()
        ax=Max_Daily_Drawdown.plot()
        for ts in mdd_idx:
            ax.axvline(ts)
    if plot:
        Daily_Drawdown.plot()
        ax=Max_Daily_Drawdown.plot()
        for ts in mdd_idx:
            ax.axvline(ts)
    ret_dict=defaultdict()
    ret_dict['mdd_ser']=Max_Daily_Drawdown
    ret_dict['mdd']= Max_Daily_Drawdown.min()
    if 'atr' in pdf.columns:
        atr=pdf['atr']
    else:
        atr=pdf.ta.atr(atr_bars)
    ret_dict['atr_exit']=Roll_Min-atr*atr_multiple
    ret_dict['atr_exit_dd']=1-ret_dict['atr_exit']/Roll_Min
    ret_dict['dd_if_exited']=(Roll_Min-pdf.Close)/Roll_Min
    ret_dict['dd_margin_to_exit']=ret_dict['atr_exit_dd']-ret_dict['mdd_ser']
    ret_dict['mdd_exceeded']=ret_dict['dd_margin_to_exit'].any()<0
    ret_dict['mdd_idx']=mdd_idx
    ret_dict['pct_ret']=(pdf.Close/pdf.Close.iloc[0])-1
    ret_dict['pct_ret_if_exited']=ret_dict['pct_ret'].loc[mdd_idx[-1]]

    return ret_dict


def backtest_between(entry_date, exit_date, price_df, rprice_df, trade_type='long', extra_info_dict={}):
    '''
    trade_type can be long, short, pair_long, pair_short
    '''
    import math
    
    #dlog(f'#backtest_between {price_df.index[-1]}, {exit_date}, {len(price_df)}, {extra_info_dict}')
    import pandas as pd
    price_df['_date']=price_df.index
    rprice_df['_date']=rprice_df.index
    rprice_df=pd.merge(rprice_df, price_df['_date'], right_index=True, left_index=True, how='outer').ffill()
    idx=(price_df.index>=entry_date)
    _=price_df[idx]
    if len(_)==0:
        dlog(f'skip cant get entry date price {entry_date, extra_info_dict}')
        return None

    if exit_date is None:
        exit_date=price_df.index[-1]
    if exit_date>price_df.index[-1]:
        exit_date_lag=price_df.index[-1]
    else:
        if len(price_df[exit_date:])>1:
            exit_date_lag=price_df[exit_date:].index[1]
        else:
            if len(price_df[exit_date:])>0:
                exit_date_lag=price_df[exit_date:].index[0]
            else:
                return None
    subdf=price_df[entry_date:exit_date_lag].copy()
    subdf.dropna(inplace=True)
    if len(subdf)==0:
        dlog(f'empty subdf for price_df {extra_info_dict, len(subdf)}')
        return None
    rsubdf=rprice_df[entry_date:exit_date_lag].copy()
    if len(rsubdf)==0:
        dlog(f'rprice_df not updated compared with price_df {extra_info_dict, len(rsubdf)}, rprice_df at {rprice_df.index[-1]} vs {exit_date_lag}')
        ridx=rprice_df.index
        raise Exception(f"rprice_df not updated {rprice_df.index[-1]} vs {entry_date} to  {exit_date_lag} {rprice_df.shape}, {rprice_df.tail(2)}, {type(_.index[-1])}, {type(ridx[-1])}")
        return None

#        price_pct_diff=(subdf.Close[-1]-subdf.Close[0])/subdf.Close[0]
#        rprice_pct_diff=(rsubdf.Close[-1]-rsubdf.Close[0])/rsubdf.Close[0]
    lag_offset=1
    rlag_offset=1
#    subdf_high=subdf.High.max()
#    subdf_low=subdf.Low.min()
    if len(subdf.Close)<2:
        lag_offset=0
        dlog(f'error with subdf {subdf, len(subdf), entry_date, exit_date_lag}')
    if len(rsubdf.Close)<2:
        rlag_offset=0
        dlog(f'error with rsubdf  {rsubdf, len(rsubdf), entry_date, exit_date_lag}')
    if len(subdf.Close)>1 and len(rsubdf.Close)>1:
        price_pct_diff_l1=(subdf.Close[-1]-subdf.Close[lag_offset])/subdf.Close[lag_offset]
        rprice_pct_diff_l1=(rsubdf.Close[-1]-rsubdf.Close[rlag_offset])/rsubdf.Close[rlag_offset]


    subdf['price_pct_diff']=(subdf.Close-subdf.Close[0])/subdf.Close[0]
    subdf['rprice_pct_diff']=(rsubdf.Close-rsubdf.Close[0])/rsubdf.Close[0]
    subdf['pair_pct_diff']=subdf.eval('price_pct_diff-rprice_pct_diff')
    subdf['price_pct_diff_l1']=(subdf.Close-subdf.Close[lag_offset])/subdf.Close[lag_offset]
    subdf['rprice_pct_diff_l1']=(rsubdf.Close-rsubdf.Close[rlag_offset])/rsubdf.Close[rlag_offset]
    subdf['pair_pct_diff_l1']=subdf.eval('price_pct_diff_l1-rprice_pct_diff_l1')


    price_pct_diff=subdf['price_pct_diff'].iloc[-1]
    rprice_pct_diff=subdf['rprice_pct_diff'].iloc[-1]
    pair_pct_diff=subdf['pair_pct_diff'].iloc[-1]
    price_pct_diff_l1=subdf['price_pct_diff_l1'].iloc[-1]
    rprice_pct_diff_l1=subdf['rprice_pct_diff_l1'].iloc[-1]
    pair_pct_diff_l1=subdf['pair_pct_diff_l1'].iloc[-1]
    ticker='n/a'
    rticker='n/a'
    mdd=0
    mdd_exceeded=False
    mdd_date=subdf.index[-1]
    ret_at_mdd=subdf.Close/subdf.iloc[-1].Close
    if 'ticker' in extra_info_dict.keys():
        ticker=extra_info_dict['ticker']
        rticker=extra_info_dict['rticker']
    if trade_type=='long':
        drawdown=min(subdf.Low.min()-subdf.Close[0], 0)
        mdd_dict=get_long_max_drawdown_details(subdf, atr_bars=22, atr_multiple=4)
        mdd_exceeded=mdd_dict['mdd_exceeded']
        mdd_date= mdd_dict['mdd_idx'][-1]
        mdd= mdd_dict['mdd']
        ret_at_mdd=mdd_dict['pct_ret_if_exited']
        _=subdf[(subdf.Low==subdf.Low.min())]
        if len(_)==0:
            dlog(f'long subdf drawdown is {subdf.Low.min(), len(subdf), subdf}')
        drawdown_date=subdf[(subdf.Low==subdf.Low.min())].index[0]
        peakprofit=max(subdf.High.max()-subdf.Close[0], 0)
        _=subdf[(subdf.High==subdf.High.max())]
        if len(_)==0:
            dlog(f'long subdf peak is {subdf.Low.min(), len(subdf)}')
        peakprofit_date=subdf[(subdf.High==subdf.High.max())].index[0]
        pct_profit=price_pct_diff
        pct_profit_l1=price_pct_diff_l1
        remarks='l_%s' % ticker
    if trade_type=='short':
        peakprofit=-1*min(subdf.Low.min()-subdf.Close[0], 0)
        mdd_dict=get_short_max_drawdown_details(subdf, atr_bars=22, atr_multiple=4)
        mdd_exceeded=mdd_dict['mdd_exceeded']
        mdd_date= mdd_dict['mdd_idx'][-1]
        mdd=mdd_dict['mdd']
        ret_at_mdd=mdd_dict['pct_ret_if_exited']
        _=subdf[(subdf.Low==subdf.Low.min())]
        if len(_)==0:
            dlog(f'short subdf peak is {subdf.Low.min(), len(subdf)}')
        peakprofit_date=subdf[(subdf.Low==subdf.Low.min())].index[0]
        drawdown=-1*max(subdf.High.max()-subdf.Close[0], 0)
        _=subdf[(subdf.High==subdf.High.max())]
        if len(_)==0:
            dlog(f'short subdf drawdown is {subdf.High.max(), len(subdf)}')
        drawdown_date=subdf[(subdf.High==subdf.High.max())].index[0]
        price_pct_diff=price_pct_diff*-1
        price_pct_diff_l1=price_pct_diff_l1*-1
        pct_profit=price_pct_diff
        pct_profit_l1=price_pct_diff_l1
        remarks='s_%s' % ticker
    if trade_type.split('_')[0]=='pair':

        pct_profit=subdf.iloc[-1]['pair_pct_diff']
        pct_profit_l1=subdf.iloc[-1]['pair_pct_diff_l1']
        ret_at_mdd=mdd_dict['pct_ret_if_exited']
        if trade_type.split('_')[1]=='long':
            remarks='l_%s|s_%s' % (ticker, rticker)
            peakprofit=subdf['pair_pct_diff'].max()
            _idx=(subdf['pair_pct_diff']==peakprofit)
            if len(subdf[_idx])==0:
                dlog(f'cant find peak profit date for {peakprofit, ticker, rticker}')
                peakprofit_date=subdf.index[0]
            else:
                peakprofit_date=subdf[_idx].index[0]
            drawdown=subdf['pair_pct_diff'].min()
            _idx=(subdf['pair_pct_diff']==drawdown)
            if len(subdf[_idx])==0:
                dlog(f'cant find peak drawndown date for {drawdown , ticker, rticker}')
                drawdown_date=subdf.index[0]
            else:
                drawdown_date=subdf[_idx].index[0]
        else:
            remarks='l_%s|s_%s' % (rticker, ticker)
            peakprofit=subdf['pair_pct_diff'].min()
            drawdown=subdf['pair_pct_diff'].max()
            _idx=(subdf['pair_pct_diff']==peakprofit)
            if len(subdf[_idx])==0:
                dlog(f'cant find peak profit date for  {peakprofit, ticker, rticker}')
                peakprofit_date=subdf.index[0]
            else:
                peakprofit_date=subdf[_idx].index[0]
            _idx=(subdf['pair_pct_diff']==drawdown)
            if len(subdf[_idx])==0:
                dlog(f'cant find drawdown date for {drawdown, ticker, rticker}')
                drawdown_date=subdf.index[0]
            else:
                drawdown_date=subdf[(subdf['pair_pct_diff']==drawdown)].index[0]
            drawdown=drawdown*-1
            peakprofit=peakprofit*-1
            pct_profit=pct_profit*-1
            pct_profit_l1=pct_profit_l1*-1


    if extra_info_dict is None:
        trade_dict=defaultdict()
    else:
        trade_dict=extra_info_dict.copy()
    trade_dict['mdd_exceeded']=mdd_exceeded
    trade_dict['mdd_date']=mdd_date
    trade_dict['ret_at_mdd']=ret_at_mdd
    trade_dict['mdd']=mdd
    trade_dict['trade_type']=trade_type
    trade_dict['rprice_pct_diff']=rprice_pct_diff
    trade_dict['price_pct_diff']=price_pct_diff
    trade_dict['pair_pct_diff']=pair_pct_diff
    trade_dict['pct_profit']=pct_profit
    trade_dict['rprice_pct_diff_l1']=rprice_pct_diff_l1
    trade_dict['price_pct_diff_l1']=price_pct_diff_l1
    trade_dict['pair_pct_diff_l1']=pair_pct_diff_l1
    trade_dict['pct_profit_l1']=pct_profit_l1
    trade_dict['r_entry_price']=rsubdf.Close[0]
    trade_dict['r_exit_price']=rsubdf.Close[-1]
    trade_dict['entry_date']=entry_date
    trade_dict['entry_price']=subdf.Close[0]
    trade_dict['exit_date']=exit_date
    trade_dict['exit_price']=subdf.Close[-1]
    trade_dict['drawdown']=drawdown/trade_dict['entry_price']
    trade_dict['drawdown_day']=drawdown_date-entry_date
    trade_dict['peakprofit']=peakprofit/trade_dict['entry_price']
    trade_dict['peakprofit_day']=peakprofit_date-entry_date
    trade_dict['day_in_trade']=max(1, (exit_date-entry_date).days)

    trade_dict['annualized_profit']=trade_dict['pct_profit']/trade_dict['day_in_trade']*365
    trade_dict['mean_return_daily']=subdf.Close.pct_change().mean()
    trade_dict['mean_return_std']=subdf.Close.pct_change().std()
    trade_dict['sharpe']=trade_dict['annualized_profit']/(trade_dict['mean_return_std']*math.sqrt(250))
    trade_dict['pct_change_std']=subdf.Close.pct_change().std()
    trade_dict['remarks']=remarks
    trade_dict['pos_scale']=1.0
    return trade_dict


def backtest_from_tradedates_df( tradedates_df, price_df, rprice_df, trade_type='long', extra_info_dict={}):
    '''
    extra_info_dict={'ticker':t, 'rticker':rticker}
    required fields for tradedates_df are just 'entry_date' and 'exit_date'
    in case we have to use signal date plus one entry date, rename them
    return trades_df which is the trades dataframe
    '''
    import pandas as pd
    all_trades=defaultdict()
    for k,v in tradedates_df.iterrows():
        tradedates_row=v.to_dict()
        entry_date=pd.to_datetime(v['entry_date'])
        exit_date=pd.to_datetime(v['exit_date'])
        #dlog(f'entry_date:{entry_date}')
        #dlog(f'{price_df.iloc[-1].index, entry_date}')
        if pd.to_datetime(entry_date)>price_df.index[-1]:
            continue
        _=price_df[entry_date:].iloc[0]
        tradedates_row['entry_price']=_.Close

        _=price_df[exit_date:]
        if  len(_)==0:
            _=price_df.iloc[-1]
        else:
            _=price_df[exit_date:].iloc[0]
        tradedates_row['exit_price']=_.Close
        for k1 in extra_info_dict:
            tradedates_row[k1]=extra_info_dict[k1]

        trade_dict=(backtest_between(entry_date, exit_date, price_df, rprice_df, trade_type=trade_type, extra_info_dict=tradedates_row))
        if trade_dict is None:
            continue
        all_trades[k]=trade_dict
    tdf=pd.DataFrame.from_dict(all_trades, orient='index')

    return tdf



class chandelierExitBacktester:
    '''
    typical use:


    verify trade:  given trade file csv, call plot_trades followed by  verify_chand_ex_trades
    '''
    @staticmethod
    def verify_chand_ex_trades(trades_df, trade_idx=0):
        '''
        verify trades read from file (can get it from plot_trades , so you have the plot also
        keywords argument:
        DataFRame trades_df: trades file to verify using backtester.get_ch_ex_trade_exit_date
        int trade_idx:  the trade row to verify with print out and chart
        return tuple of trades_df and ax to draw on

        '''
        subdf=show_trades(trades_df).copy()
        dlog(subdf.entry_date)
        subdf=subdf.iloc[trade_idx:trade_idx+1]
        ticker=subdf.iloc[0].ticker
        from backtest.chandelierExitBacktester import chandelierExitBacktester as backtester
        from datalib.commonUtil import commonUtil as cu
        pdf=cu.read_quote(ticker)
        entry_date=subdf.iloc[0].entry_date
        exit_date=subdf.iloc[0].exit_date


        tradedates_df=backtester.get_ch_ex_trade_exit_date(subdf, pdf, ticker, plot=True)


        ax=pdf.loc[entry_date:exit_date].Low.plot()
        return tradedates_df, ax

    @staticmethod
    def plot_trades(ticker='GBTC', trades_csvname=None):
        '''
        plot the trades from the trade_csvname, or guess the file name from
           f'results/{ticker}.catboost._lb_7_spikeup.long.trades.csv'
        return the trades data frame
        '''
        import pandas as pd
        from datalib.commonUtil import commonUtil as cu

        if trades_csvname is None:
            trades_csvname=f'results/{ticker}.catboost._lb_7_spikeup.long.trades.csv'

        tdf=pd.read_csv(trades_csvname, index_col=0)
        pdf=cu.read_quote(ticker)
        dlog(tdf.entry_date)
        pdf.loc[tdf.entry_date].Close

        cols=['Close']
        #pdf.loc[tdf.entry_date].Close
        ax=pdf[cols].iloc[-80:].plot(figsize=(12,7))
        ax.scatter(x=tdf.entry_date, y=pdf.loc[tdf.entry_date].Close, marker='^', s=180, color='g')
        ax.scatter(x=tdf.exit_date, y=pdf.loc[tdf.exit_date].Close, marker='v', s=150, color='r')
        ax.set_title(f'{ticker} entry:green      exit:red')
        for idx in tdf.exit_date:
            print('exit:', idx)
            ax.axvline(idx, color='r')

        return tdf

    @staticmethod
    def get_chandelier_long_exit_signal(pdf, atr_bars=22, retrace_atr_multiple=3, def_pct_stop=0.1, smooth_bars=6, ret_details=False, plot=False):
        '''
        note that this function require the 'atr' column be set before passing in
        Keyword Argruments:
        DataFrame pdf: price_df from commonUtil.read_quote, need to have the atr columns set
                before otherwise some nan value will appear in the start of the series
        int retrace_atr_multiple: no of atr multiple retracement before exit
        int smooth_bars: rolling max for the output signal, so that the exist signal
                remain high after the cross over, for use in breadth calcuation
        '''
        
        ret_df=pdf
        _high=pdf.High.rolling(atr_bars, min_periods=2).max().shift(1).bfill()
        _low=pdf.Low.rolling(int(atr_bars/2), min_periods=2).min().shift(1).bfill()
        if 'atr' in pdf.columns:
            atr=pdf['atr']
        else:
            import pandas_ta
            atr=pdf.ta.atr(atr_bars).shift(1)
            pdf['atr']=atr
            ret_df['atr']=atr
        ret_df['atr']=atr
        # ch_ex_h is the chandelier exit of recent High - n*atr
        # ch_ex_l is the default dynamic stoploss which is the recent low - 1*atr
        ch_ex_h=_high-atr*retrace_atr_multiple
        ch_ex_l=_low-atr*1

        mdd_dict=get_long_max_drawdown_details(pdf,day_cnt=250, plot=plot)
        ret_df['long_max_drawdown']=mdd_dict['mdd']
        ret_df['ch_ex_h']=ch_ex_h
        ret_df['ch_ex_l']=ch_ex_l
        entry_price=pdf.Close.iloc[0]
        ret_df['entry_stop']=entry_price*(1-def_pct_stop)
        filter_idx=(entry_price< ret_df['ch_ex_h'])*1.0  # if the high chandelier exist is > entry price, adjust it to zero to give way to default / static atr exit
        ret_df['ch_ex_h2']=ret_df['ch_ex_h']*filter_idx
        # finally the actaully chandelier exit is the high of the default retracement exit, static $def_pct_stop exit and the trend following recent high - 3 * atr exit
        ret_df['chand_ex']=ret_df[['ch_ex_l', 'ch_ex_h2', 'entry_stop']].max(axis=1).rolling(10, min_periods=1).max()
        ret_df['low']=pdf.Low
        ret_df['close']= pdf.Close
        ret_df['exit_signal']=get_signal_cross(ret_df, 'low', 'chand_ex')<0

        if plot:
#            sub_cols=['ch_ex_h', 'ch_ex_l', 'chand_ex', 'low' ]
            sub_cols=['chand_ex', 'low' ]
    #            sub_cols=['close', 'low']
            plot_df=(ret_df[sub_cols]).dropna()
            if len(plot_df)>0:
                ax=ret_df[sub_cols].plot()
                bidx=ret_df['exit_signal'].dropna()>0
                idx_list=list(ret_df[bidx].index)
                #dlog(plot_df)
                #dlog(f'idx_list {idx_list}')
                ax.scatter(x=idx_list, y=plot_df.loc[idx_list].low, marker='^', s=180, color='g')
                ax.scatter(x=idx_list, y=plot_df.loc[idx_list].low, marker='^', s=180, color='g')
                for idx in idx_list:
#                    dlog(f'exit trade at {idx}', ret_df.loc[:idx].iloc[-5:])
                    ax.axvline(idx)


        if ret_details:
            ret_df['exit_signal']=ret_df['exit_signal'].rolling(smooth_bars).max()
            return 'ch_ex20d', ret_df
        return 'ch_ex20d', ret_df['exit_signal'].rolling(smooth_bars).max()


    @staticmethod
    def get_ch_ex_trade_exit_date(entry_date_df, pdf, ticker, trade_type='long', ex_atr_bars=22, atr_multiple=3, def_pct_stop=0.10, signame='external', smooth_bars=1, rticker='SPY', plot=False):
        '''
        entry_date_df requires ticker, entry_date columns
        if no entry_date but has signal_date, it will create entry_date column with signal_date+1 trading day
        note that no matter long of short, signal>0 mean exit, which is already accounted for in get_chandelier_long_exit_signal
        '''
        import pandas_ta
        data_rows=defaultdict()

#        ret_df=pd.DataFrame()
#        dlog(f'entry_date_df columns:{entry_date_df.columns}')
        if 'entry_date' not in entry_date_df.columns:
            entry_date_df['entry_date']=entry_date_df['signal_date'].apply(lambda x:get_next_day(x, pdf))
        first_entry_date=entry_date_df['entry_date'].iloc[0]

        pdf['atr']=pdf.ta.atr(ex_atr_bars)

        signame, _=chandelierExitBacktester.get_chandelier_long_short_exit_signal(pdf.loc[first_entry_date:].copy(), trade_type=trade_type, atr_bars=ex_atr_bars, def_pct_stop=def_pct_stop, retrace_atr_multiple=atr_multiple,  smooth_bars=smooth_bars, plot=plot)
        exit_idx=_>0
        exit_cross=_[exit_idx]

        if not 'signame' in entry_date_df:
            entry_date_df['signame']=signame
        for v,r in entry_date_df.iterrows():
            row=defaultdict()
            entry_date=r['entry_date']
            ticker=r['ticker']
            subdf=pdf.loc[entry_date:].copy()
            entry_price=pdf.loc[entry_date].Close
            _=exit_cross.loc[entry_date:].copy()
            if len(_)==0:
                exit_date=subdf.index[-1]
            else:
                exit_date=_.index[0]
            signame=r['signame']
            row['trade_type']=trade_type
            row['ticker']=ticker
            row['rticker']=rticker
            row['entry_date']=entry_date
            row['exit_date']=exit_date
            row['entry_price']=entry_price
            data_rows['%s_%s' % (ticker, entry_date)]=row

        tradedates_df=pd.DataFrame.from_dict(data_rows).T
        return tradedates_df

    @staticmethod
    def summarise_trades(all_trade_df, plot=False, long_short='both', cutoff_date=None)->dict:
        '''
        Keyword Arguments:
        DataFrame all_trade_df: with all backtested trades return ratio, drawdown etc
        long_short = 'long', 'short', 'both'
        Dateime cutoff_date: only take trades up to that date
        Return:
            dict with all benchmark summaries

        '''
        ret_dict=defaultdict()
        if cutoff_date is None:
            _all_trade_df=all_trade_df.copy()
        else:
            _all_trade_df=all_trade_df.query('entry_date < %s' % cutoff_date)

        if long_short != 'both':
            _all_trade_df=all_trade_df.query('trade_type=="%s"' % long_short).copy()
        trade_type=all_trade_df.iloc[-1]['trade_type']
        rticker=all_trade_df.iloc[-1]['rticker']
        ret_dict['trade_type']=trade_type
        ret_dict['rticker']=rticker
        if 'mdd' in all_trade_df.columns:
            ret_dict['mean_max_drawdown']=all_trade_df.mdd.mean()

        ret_dict['mean_drawdown']=all_trade_df.drawdown.mean()
        ret_dict['median_drawdown']=all_trade_df.drawdown.mean()
        ret_dict['mean_drawdown_day']=all_trade_df.drawdown_day.mean()
        ret_dict['median_drawdown_day']=all_trade_df.drawdown_day.mean()
        ret_dict['mean_peakprofit']=all_trade_df.peakprofit.mean()
        ret_dict['median_peakprofit']=all_trade_df.peakprofit.mean()
        ret_dict['mean_peakprofit_day']=all_trade_df.peakprofit_day.mean()
        ret_dict['median_peakprofit_day']=all_trade_df.peakprofit_day.mean()
        ret_dict['mean_day_in_trade']=all_trade_df.day_in_trade.mean()
        ret_dict['median_day_in_trade']=all_trade_df.day_in_trade.median()

        ret_dict['mean_pct_profit']=all_trade_df.pct_profit.mean()
        ret_dict['median_pct_profit']=all_trade_df.pct_profit.median()
        ret_dict['max_trade_pct_profit']=all_trade_df.pct_profit.max()
        ret_dict['all_trades']=all_trade_df
        ret_dict['best_trade']=all_trade_df.query('pct_profit==%s' % ret_dict['max_trade_pct_profit']).iloc[0]
        ret_dict['best_ticker']=ret_dict['best_trade']['ticker']
        ret_dict['best_entry']=ret_dict['best_trade']['entry_date']
        ret_dict['best_exit']=ret_dict['best_trade']['exit_date']
        ret_dict['best_pct_profit']=ret_dict['best_trade']['pct_profit']
        ret_dict['best_trade_drawdown']=ret_dict['best_trade']['drawdown']
        ret_dict['max_trade_pct_loss']=all_trade_df.pct_profit.min()
        ret_dict['worst_trade']=all_trade_df.query('pct_profit==%s' % ret_dict['max_trade_pct_loss']).iloc[0]
        ret_dict['worst_ticker']=ret_dict['worst_trade']['ticker']
        ret_dict['worst_entry']=ret_dict['worst_trade']['entry_date']
        ret_dict['worst_exit']=ret_dict['worst_trade']['exit_date']
        ret_dict['worst_pct_profit']=ret_dict['worst_trade']['pct_profit']
        ret_dict['worst_drawdown']=ret_dict['worst_trade']['drawdown']

        ret_dict['sum_pct_profit']=all_trade_df.pct_profit.sum()
        ret_dict['trade_cnt']=len(all_trade_df)
        ret_dict['win_trade_cnt']=len(all_trade_df.query('pct_profit>0'))
        ret_dict['win_trade_avg_pct_profit']=all_trade_df.query('pct_profit>0').pct_profit.mean()
        ret_dict['win_trade_total_profit']=all_trade_df.query('pct_profit>0').pct_profit.sum()
        ret_dict['win_trade_median_pct_profit']=all_trade_df.query('pct_profit>0').pct_profit.median()
        ret_dict['win_trade_median_day_in_trade']=all_trade_df.query('pct_profit>0').day_in_trade.median()
        ret_dict['lose_trade_cnt']=len(all_trade_df.query('pct_profit<0'))
        ret_dict['lose_trade_avg_pct_loss']=all_trade_df.query('pct_profit<0').pct_profit.mean()
        ret_dict['lose_trade_total_pct_loss']=all_trade_df.query('pct_profit<0').pct_profit.sum()
        ret_dict['lose_trade_median_pct_loss']=all_trade_df.query('pct_profit<0').pct_profit.median()
        ret_dict['lose_trade_median_day_in_trade']=all_trade_df.query('pct_profit<0').day_in_trade.median()
        ret_dict['start_entry']=all_trade_df.entry_date.min()
        ret_dict['end_exit']=all_trade_df.exit_date.max()
        ret_dict['duration']=(ret_dict['end_exit']-ret_dict['start_entry']).days+0.5
        ret_dict['total_day_in_trade']=all_trade_df.day_in_trade.sum()
        ret_dict['exposure_portion']=ret_dict['total_day_in_trade']/ret_dict['duration']
        ret_dict['annualized_gain']=ret_dict['sum_pct_profit']*365/ret_dict['duration']
        ret_dict['exposure_adjusted_annualized_gain']=ret_dict['annualized_gain']/ret_dict['exposure_portion']
        ret_dict['exposure_adjusted_annualized_gain_per_mean_drawdown']=ret_dict['exposure_adjusted_annualized_gain']/abs(ret_dict['mean_drawdown'])
        if plot:
            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(1, 2, sharey=True, sharex=False, tight_layout=True)
            axs[0].hist(all_trade_df.query('pct_profit>0').pct_profit, bins=30)
            axs[0].set_title('winner pct_profit')
            axs[1].hist(all_trade_df.query('pct_profit<0').pct_profit, bins=30)
            axs[1].set_title('loser pct_profit')

            fig, axs = plt.subplots(1, 2, sharey=False, sharex=False, tight_layout=True)
            axs[0].hist(all_trade_df.query('pct_profit>0').day_in_trade, bins=30)
            axs[0].set_title('winner day in trade')
            axs[1].hist(all_trade_df.query('pct_profit<0').day_in_trade, bins=30)
            axs[1].set_title('loser day in trade')

        return ret_dict

    @staticmethod
    def gen_ch_ex_trades_from_tradedates(tradedates_df, price_df, ticker, trade_type='long', rprice_df=None, rticker='SPY', atr_multiple=3)->tuple:
        '''
        Keyword arguments:
            DataFrame tradedates_df: from  get_ch_ex_trade_exit_date which take entry_date columns add chand_ex signal and complete with exit_date
            DataFrame rprice_df: reference benchmark like SPY quote, for pair trades
        Returns:
            tutple summary_dict, trades_df
        '''

        if rprice_df is None:
            rprice_df=cu.read_quote(rticker)

        trades_df=backtest_from_tradedates_df(tradedates_df, price_df, rprice_df, trade_type=trade_type)
#        dlog(f'trades_df cols {trades_df.columns, trades_df.shape}')
        if len(trades_df)==0:
            return None, None
        trades_df['ticker_']=ticker
        trades_df['entry_date_']=trades_df['entry_date']
        trades_df['params']=f'atr:{atr_multiple}'
        trades_df.set_index(['ticker_', 'entry_date_'], inplace=True)
        summary_dict=chandelierExitBacktester.summarise_trades(trades_df, long_short=trade_type)
        return summary_dict, trades_df

    @staticmethod
    def add_backtest_from_pred_df(ticker, pred_df, pred_col='pred_y', rticker='SPY', signame='catboost._lb_spikeup',
            trade_type='long', retrace_atr_multiple=3, ex_atr_bars=20, def_pct_stop=0.1, plot=False):
        '''

        Keyword arguments:
            str ticker: the ticker of security uin concnern
            DataFrame pred_df: df from the truth table returned by the ML advistor, using has columns ['y', 'pred_y']
            str pred_col: indicate which column to use for prediction
            str rticker: reference ticker in pair trade
            str signame: index to use for each row in batch advistor worflow
            str trade_type: long/short/pair_long/pair_short
            int retrace_atr_multiple: the atr multiple used for retracement

        Return dict:
            ret_dict.keys() contains ['all_trades_df', 'all_tradesummary', 'new_res_df']
        '''
        entry_date_df=pred_df.query(f'{pred_col}>0').copy()
        if len(entry_date_df)==0:
            return None
        entry_date_df['entry_date']=entry_date_df.index
        entry_date_df['rticker']=rticker
        if not ticker is None:
            entry_date_df['ticker']=ticker
    #        ticker=list(pred_df.ticker.unique())
        entry_date_df['signame']=signame
        price_df=cu.read_quote(ticker)
        tradedates_df=chandelierExitBacktester.get_ch_ex_trade_exit_date(entry_date_df, price_df, ticker=ticker,
                ex_atr_bars=ex_atr_bars, trade_type=trade_type, atr_multiple=retrace_atr_multiple, rticker=rticker,
                def_pct_stop=def_pct_stop,  plot=plot)

        tradesummary, trades_df=chandelierExitBacktester.gen_ch_ex_trades_from_tradedates(tradedates_df, price_df, ticker=ticker, rticker=rticker, trade_type=trade_type)
        if not trades_df is None:
           trades_df['signame']=signame
        ret_dict=defaultdict()
        ret_dict['all_tradesummary']=tradesummary
        ret_dict['trades_df']=trades_df
        return ret_dict

    @staticmethod
    def add_labels_by_bars(ndf, bars=15, threshold=0.09):
        '''
        create _lb_{bars}_labels according to the threshold,
        spikeup mean reached threshold at least once,
        bigrise mean still above threshold after x bars
        '''
        if len(ndf)<100:
            dlog('insufficent data')
            return None

        fieldname='_n_ret%s' % bars
        ndf[fieldname]=(ndf.Close.shift(-bars)-ndf.Close)/ndf.Close
        ndf[f'_lb_{bars}_bigrise']=ndf[fieldname]>threshold
        ndf[f'_lb_{bars}_bigdrop']=ndf[fieldname]<-threshold
        ndf[f'_lb_{bars}_bigrise']=ndf[f'_lb_{bars}_bigrise']*1
        ndf[f'_lb_{bars}_bigdrop']=ndf[f'_lb_{bars}_bigdrop']*1

        low_fieldname=f'_n_{bars}d_low'
        high_fieldname=f'_n_{bars}d_high'
        ndf[low_fieldname]=ndf.Low.shift(-bars).rolling(window=(bars),min_periods=3).min()
        ndf[high_fieldname]=ndf.High.shift(-bars).rolling(window=(bars),min_periods=3).max()
#        dlog(f'low vs high file:{low_fieldname}, {high_fieldname}', ndf.tail())
        ndf[f'_lb_{bars}_spikedown']=ndf.eval(f'(Close  - {low_fieldname})/Close')>threshold
        ndf[f'_lb_{bars}_spikedown']=ndf[f'_lb_{bars}_spikedown']*1
        ndf[f'_lb_{bars}_spikeup']=ndf.eval(f'({high_fieldname}-Close)/Close')>threshold
        ndf[f'_lb_{bars}_spikeup']=ndf[f'_lb_{bars}_spikeup']*1
        add_3way_label(ndf, uplabel_col=f'_lb_{bars}_bigrise', dnlabel_col=f'_lb_{bars}_bigdrop', newcol=f'_lbm_{bars}_bigmove')
        dlog(ndf[f'_lbm_{bars}_bigmove'].describe())
        add_3way_label(ndf, uplabel_col=f'_lb_{bars}_spikeup', dnlabel_col=f'_lb_{bars}_spikedown', newcol=f'_lbm_{bars}_bigspike')
        dlog(ndf[f'_lbm_{bars}_bigspike'].describe())

        label_cols=[x for x in ndf.columns if x[:3]=='_lb']
        return label_cols
    
    @staticmethod
    def get_chandelier_long_short_exit_signal(pdf, atr_bars=22, trade_type='long', retrace_atr_multiple=4, def_pct_stop=0.1, smooth_bars=6, ret_details=False, plot=False):
        '''
        note that this function require the 'atr' column be set before passing in
        Keyword Argruments:
        DataFrame pdf: price_df from commonUtil.read_quote, need to have the atr columns set
                before otherwise some nan value will appear in the start of the series
        int retrace_atr_multiple: no of atr multiple retracement before exit
        int smooth_bars: rolling max for the output signal, so that the exist signal
                remain high after the cross over, for use in breadth calcuation
        '''
        if 'atr' in pdf.columns:
            atr=pdf['atr'].bfill()
        else:
            import pandas_ta
            atr=pdf.ta.atr(atr_bars).shift(1).bfill()
            pdf['atr']=atr

        ret_df=pdf
        ret_df['atr']=atr        
        ret_df['low']=pdf.Low
        ret_df['high']=pdf.High       
        ret_df['close']= pdf.Close
        
        if trade_type=='long':
            rolling_ref=pdf.High.rolling(atr_bars, min_periods=2).max().shift(1).bfill()
            ch_ex_h=rolling_ref-atr*retrace_atr_multiple
            ch_ex_l=pdf.Low.rolling(2, min_periods=1).min().shift(2)
            mdd_dict=get_long_max_drawdown_details(pdf,day_cnt=100, plot=plot)
            ret_df['long_max_drawdown']=mdd_dict['mdd']
            ret_df['ch_ex_h']=ch_ex_h
            ret_df['ch_ex_l']=ch_ex_l
            ret_df['exit_signal_atr']=ret_df.low<ch_ex_h
            ret_df['exit_signal_prev_sup']=ret_df.low<ch_ex_l            
            ret_df['chand_ex']=ret_df['exit_signal_atr']*ch_ex_h + ~ret_df['exit_signal_atr']*ch_ex_l
            ret_df['exit_signal']=ret_df['low']<ret_df['chand_ex']
            cols=['low', 'rolling_ref', 'chand_ex']
            ret_df['pos_idx']=~(ret_df['exit_signal_prev_sup']+ret_df['exit_signal_atr'])
            marker='v'
            acolor='r'            
            sig_col='low'
            pos_color='lightgray'

        else:
            rolling_ref=pdf.Low.rolling(atr_bars, min_periods=2).min().shift(1).bfill()
            ch_ex_l=rolling_ref+atr*retrace_atr_multiple
            ch_ex_h=pdf.High.rolling(2, min_periods=1).max().shift(2)
            mdd_dict=get_short_max_drawdown_details(pdf,day_cnt=100, plot=plot)
            ret_df['short_max_drawdown']=mdd_dict['mdd']
            ret_df['ch_ex_h']=ch_ex_h
            ret_df['ch_ex_l']=ch_ex_l
            ret_df['exit_signal_atr']=ret_df.high>ch_ex_h
            ret_df['exit_signal_prev_sup']=ret_df.high>ch_ex_l            
            ret_df['chand_ex']=ret_df['exit_signal_atr']*ch_ex_l + ~ret_df['exit_signal_atr']*ch_ex_h           
            ret_df['exit_signal']=ret_df['high']>ret_df['chand_ex']
            ret_df['pos_idx']=ret_df['chand_ex']>ret_df['high']
            neg_idx=~ret_df['pos_idx']
            cols=['high', 'rolling_ref', 'chand_ex']
            sig_col=['high']
            marker='^'
            acolor='g'
            pos_color='lightgreen'            

        ret_df['rolling_ref']=rolling_ref
        entry_price=pdf.Close.iloc[0]
        ret_df[cols].plot(figsize=(10,8))

        if plot:
            plot_df=(ret_df[cols]).dropna()
            if len(plot_df)>0:
                ax=ret_df[cols].plot()
                ret_df['rolling_ref'].plot(style='--', color='black')
                bidx=ret_df['exit_signal'].dropna()>0
                idx_list=list(ret_df[bidx].index)
                #dlog(plot_df)
                #dlog(f'idx_list {idx_list}')
                ax.scatter(x=idx_list, y=plot_df.loc[idx_list][sig_col], marker=marker, s=180, color=acolor)
                for idx in idx_list:
#                    dlog(f'exit trade at {idx}', ret_df.loc[:idx].iloc[-5:])
                    ax.axvline(idx)
                bidx2=ret_df['pos_idx']>0
                idx_list2=list(ret_df[bidx2].index)
                print('idx_list2 len:',len(idx_list2))
                for idx in idx_list2:
                    ax.axvline(idx, color=pos_color, zorder=0)



        if ret_details:
            ret_df['exit_signal']=ret_df['exit_signal'].rolling(smooth_bars).max()
            return 'ch_ex20d', ret_df
        return 'ch_ex20d', ret_df['exit_signal'].rolling(smooth_bars).max()
    
def test():    
    #%matplotlib inline
    ticker='ARKK'
    cu.download_quote(ticker)
    pdf=cu.read_quote(ticker)
    pdf=pdf.iloc[-550:-250]
    chandelierExitBacktester.get_chandelier_long_short_exit_signal(pdf, trade_type='long', plot=True)
    pdf=pdf.iloc[-250:]
    pdf=pdf.iloc[-250:]
    chandelierExitBacktester.get_chandelier_long_short_exit_signal(pdf, trade_type='short', plot=True, retrace_atr_multiple=3)
    
if __name__=='__main__':
    test()    
