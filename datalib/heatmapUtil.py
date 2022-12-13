from datalib.commonUtil import commonUtil as cu
from backtest.chandelierExitBacktester import dlog, showtable
import pandas as pd
import dataframe_image as dfi

class heatmapUtil:
    @staticmethod
    def time_matrix_as_heatmap(time_matrix_df, skip_cnt=10, view_cnt=30, imgofname=None, focus=None):
        '''
        trim and sample a multi columns time series table and present as heatmap over typlej
        Keyword arguments:
            DataFrame time_matrix_df : time series table from  get_rel_nday_ma_zscore_heatmap or  get_time_matrix_ranked
            int skip_cnt: [::{skip_cnt}] in sampling the time series
            int view_cnt: show the last {view_cnt} of the  timeseries
        Returns:
            html styler for display in jupyter
        '''
        def time_matrix_to_heat_df(time_matrix_df, skip_cnt=10, view_cnt=30):
            subidx=list(time_matrix_df.index)
            subidx2=subidx[::-1][::skip_cnt][::-1][-view_cnt:]
            sort_col=subidx2[-1]
            heat_df=time_matrix_df.loc[subidx2].ffill().T.sort_values(by=sort_col, ascending=False)
            return heat_df
        heat_df=time_matrix_to_heat_df(time_matrix_df, skip_cnt, view_cnt)
        heat_df2=heatmapUtil.trim_heatmap_df_for_display(heat_df, max_cols=view_cnt)
        styler=heat_df2.style.background_gradient(cmap='Blues')
        showtable(styler)
        if not imgofname is None:
            try:
                dfi.export(styler, imgofname, table_conversion='chrome')
            except:
                dfi.export(styler, imgofname, table_conversion='matplotlib')
            render_heatmap_with_seaborn(time_matrix_df, focus='IWM', skip=2, ofname=imgofname)

        return styler

    @staticmethod
    def trim_heatmap_df_for_display(heat_df, trim_date_columns=True, max_cols=20, sort_latest=True):
        def round_heat_df(heat_df):
            import math
            x=heat_df.abs().max().max()*100
            pw_add=2-int(math.log(x, 10))
            pw=pw_add+2
            new_df=(heat_df*(10**pw)).ffill().round()
            for c in new_df.columns:
                try:
                    new_df[c]=new_df[c].astype(int)
                except Exception as e:
                    print('exception converting heatmap value to int ',e)
                    continue
            return new_df
        if trim_date_columns:
            print('heat_df.columns:', heat_df.columns)
            rename_dict={col:pd.to_datetime(col).strftime('%m%d') for col in heat_df.columns}
            heat_df.rename(columns=rename_dict, inplace=True)
        keep_col=list(heat_df.columns)[-max_cols:]
        sort_col=list(heat_df.columns)[-1]
        if sort_latest:
            heat_df.sort_values(by=sort_col, ascending=False, inplace=True)
        return round_heat_df(heat_df[keep_col])

def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

def get_time_matrix_ranked(time_matrix_df):
    '''
    loop thru each row (datetime as index) of the matrix and apply rank to the row to get the current ranking
    Keyword arguments:
        DataFrame time_matrix_df:  df with ticker as columns,  datetime as index
    Returns: rank_df
        DataFrame with each row ranked for use with other ML function
    '''
    import pandas as pd
    rank_df=pd.DataFrame()
    all_rows={}
    for k in time_matrix_df.index:
        y=time_matrix_df.loc[k].rank()
        #rank_df[k]=y
        all_rows[k]=y
    rank_df=pd.DataFrame().from_dict(all_rows, orient='columns')
    return rank_df.T.copy()

def get_rel_nday_ma_zscore_heatmap(tlist, focus=None, list_tag='rank_sector_etf', startdate='20140101', nday=5, use_rank=False, zdays=20, skip_cnt=5, pred_date=None,  imgofname=None):
    all_dict={}
    print(tlist)
    for t in tlist:
        pdf=cu.read_quote(t, pred_date=pred_date).loc[startdate:]
        retxma=pdf.Close/pdf.Close.rolling(nday).mean()
        if zdays>0:
            all_dict[t]=zscore(retxma, window=zdays)
        else:
            all_dict[t]=retxma

    big_matrix_df=pd.DataFrame(all_dict)

    if use_rank:
        big_matrix_df=get_time_matrix_ranked(big_matrix_df)
    showtable(big_matrix_df.tail())
    heatmapUtil.time_matrix_as_heatmap(big_matrix_df, focus=focus, skip_cnt=skip_cnt, view_cnt=30,  imgofname=imgofname)
    return big_matrix_df

def get_ret_zscore_heatmap(tlist, list_tag='rank_sector_etf', startdate='20140101', interval=5, focus=None, use_rank=False, zdays=20, skip_cnt=5,  imgofname=None):
    all_dict={}
    print(tlist)
    for t in tlist:
        pdf=cu.read_quote(t).loc[startdate:]
        ret5d=pdf.Close.pct_change(interval)
        if zdays>0:
            all_dict[t]=zscore(ret5d, window=zdays)
        else:
            all_dict[t]=ret5d
    big_matrix_df=pd.DataFrame(all_dict)
    if use_rank:
        big_matrix_df=get_time_matrix_ranked(big_matrix_df)
    dlog(big_matrix_df.tail())
    heatmapUtil.time_matrix_as_heatmap(big_matrix_df, skip_cnt=skip_cnt, view_cnt=30, imgofname=imgofname)
    return big_matrix_df

def render_heatmap_with_seaborn(rank_df, pdf=None, focus='META', ofname='snshm.jpg', skip=2, row_cnt=100):
    from datalib.commonUtil import commonUtil as cu
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas_ta

    idx=rank_df.iloc[-row_cnt:].index[::skip]
    if not focus is None:
        pdf=cu.read_quote(focus)        
        pdf['hma20']=pdf.ta.hma(20)    
        pdf['hma40']=pdf.ta.hma(40)    
        scaler=pdf.Volume.max()/pdf.Close.max()
        pdf['volumec']=pdf.Volume*scaler/2

        subdf=pdf.loc[idx[0]:]
    t_order2=list(rank_df.iloc[-1].sort_values(ascending=False).index)
    t_order=t_order2
    if focus in t_order2:
        t_order2.remove(focus)
        t_order=[focus]+t_order2
    x_labels=[' ']*len(idx)
    x_labels[::4]=[v.strftime('%Y%m%d') for v in list(idx)[::4]]
    if pdf is None:
        fig, ax_list = plt.subplots(1, 1, sharex=False, figsize=(16,10))
        sns.heatmap(rank_df[t_order].loc[idx].T, xticklabels=x_labels, ax=ax_list, cbar=False)
        
    else:
        fig, ax_list = plt.subplots(2, 1, sharex=False, figsize=(16,10), gridspec_kw={'height_ratios': [1, 4]})
        sns.heatmap(rank_df[t_order].loc[idx].T, xticklabels=x_labels, ax=ax_list[1], cbar=False)
        plt.subplots_adjust(wspace=0.05, hspace=0.01)
        data=subdf
        cols=['Close','hma20', 'hma40']
        sns.lineplot(data=subdf[cols], ax=ax_list[0])   
    #    sns.barplot(data=subdf['Volume'], ax=ax_list[0])
#        sns.barplot(data=subdf['volumec'], ax=ax_list[0])

    plt.savefig(ofname)
    return ofname
