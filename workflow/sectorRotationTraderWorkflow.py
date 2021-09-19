''' this module supply standard arguments to call to catboostTradeAdvisor
     so user just need to provide high level like thresholds move (e.g. 10% in 7 days)
    default stop loss def_stop_pct and focus_etf_list to get training results '''
from datalib.catboostTradeAdvisor import catboostTradeAdvisor
from backtest.chandelierExitBacktester import dlog
def batch_sector_rotation_learning(rank_df, focus_etf_list=['CURE', 'TECL'], th=0.08,
            def_pct_stop=0.1, feat_cols=['TLT', 'RYE', 'RTM', 'EQAL'] ):
    """
    using a time-series rank dataframe to predict movement of tickers in focus_etf_list
       a wrapper function called to gen_ticker_rank_catboost_results
    Keyword arguments:
        :DataFrame rank_df: return from heatmapUtil.py get_rel_nday_ma_zscore_heatmap
        :list focus_etf_list: the list of ticker whose return we would like to forecast
        :float th: the threshold
        :float def_pct_stop: default stop level, 0.1 mean cut loss after 10% down frome ntry
        :list  feat_cols: tickers whose price action will be used as features
    Returns: tuble
        where all_tb are the performance summary and all_trades_list are all the trades suggested

    """

    import pandas as pd
    table_list=[]
    all_trades_list=[]

    for t in focus_etf_list:
        if t not in feat_cols:
            if not t in rank_df.columns:
                continue
            print('now in test getting in batch sector ',t)
            perf_df, trades_df=catboostTradeAdvisor.gen_ticker_rank_catboost_results(rank_df ,
                    focus_ticker=t, th=th, ex_atr_days=20, feat_cols=feat_cols, def_pct_stop=def_pct_stop)
            dlog(perf_df)
            if not perf_df is None:
                table_list.append(perf_df.T)
            if not trades_df is None:
                all_trades_list.append(trades_df)
    if len(table_list)>0:
        all_tb=pd.concat(table_list)

        all_trades_df=pd.concat(all_trades_list)
        return all_tb, all_trades_df
    else:
        return None, None

def review_perf(all_perf_table):
    '''
    present performance table without the details / debug field
    '''
    all_perf_table['signame']=all_perf_table.index
    all_perf_table['signame']=all_perf_table['signame'].apply(lambda x:x.split('.')[2])
    cols=['start_entry', 'end_exit', 'sum_pct_profit', 'mean_max_drawdown', 'mean_drawdown',
            'mean_pct_profit', 'exposure_adjusted_annualized_gain','lose_trade_cnt',
            'win_trade_cnt', 'total_day_in_trade','median_day_in_trade']


    #all_perf_table2.query('signame=="_lbmx_7_bigspike"')[cols]
    dlog('mean exposure adjusted return:', all_perf_table.query('signame=="_lb_7_spikeup"').exposure_adjusted_annualized_gain.mean())
    dlog('mean of all better model exposure adjusted return:', all_perf_table.query('signame=="_lb_7_spikeup" and model_best_iteration>300').exposure_adjusted_annualized_gain.mean())
    dlog('mean exposure adjusted return per drawdown:', all_perf_table.query('signame=="_lb_7_spikeup" ').exposure_adjusted_annualized_gain_per_mean_drawdown.mean())
    dlog('mean of all better model exposure adjusted return per drawdown:', all_perf_table.query('signame=="_lb_7_spikeup" and model_best_iteration>300').exposure_adjusted_annualized_gain_per_mean_drawdown.mean())
    dlog('sum of all trade return:', all_perf_table.query('signame=="_lb_7_spikeup"').sum_pct_profit.sum())
    dlog('mean of all trade return:', all_perf_table.query('signame=="_lb_7_spikeup"').sum_pct_profit.mean())
    dlog('sum of all better model trade return:', all_perf_table.query('signame=="_lb_7_spikeup" and model_best_iteration>300').sum_pct_profit.sum())
    dlog('mean of all better model trade return:', all_perf_table.query('signame=="_lb_7_spikeup" and model_best_iteration>300').sum_pct_profit.mean())
    dlog(all_perf_table.query('signame=="_lb_7_spikeup"')[cols])
    return all_perf_table.query('signame=="_lb_7_spikeup"')[cols]


def get_bull_etf_tickers():
    '''
    return default list of bull etf
    '''
    bull_lev_etf_list=['SPXL', 'UDOW', 'BNKU', 'UPRO', 'UMDD', 'TMF', 'DRN', 'ERX', 'TNA', 'UCO', 'TECL', 'FAS', 'SOXL']
    return bull_lev_etf_list

def run_bull_etf_test(th=0.1, focus_etf_list=None):
    '''
    th is the threshold to be passed to catboostTradeAdvisor for labelling
    '''
    import pandas as pd
    if focus_etf_list is None:
        focus_etf_list=get_bull_etf_tickers()
    all_perf_table, all_trades_df= run_test(th=th, focus_etf_list=focus_etf_list)
    if not all_perf_table is None:
        review_perf(all_perf_table)
    dlog(pd.to_datetime('today'))
    return all_perf_table, all_trades_df

def get_bear_etf_tickers():
    '''
    return default list of bear etf
    '''
    bear_lev_etf_list=['SMDD', 'TECS', 'SDOW', 'EDZ', 'FAZ', 'BNKD', 'SOXS', 'TZA']
    return bear_lev_etf_list

def run_bear_etf_test(th=0.1, focus_etf_list=None):
    '''
    th is the threshold to be passed to catboostTradeAdvisor for labelling
    '''
    import pandas as pd
    if focus_etf_list is None:
        focus_etf_list=get_bear_etf_tickers()
    all_perf_table, all_trades_df= run_test(th=th, focus_etf_list=focus_etf_list)
    if not all_perf_table is None:
        review_perf(all_perf_table)
    return all_perf_table, all_trades_df

def run_test(th=0.10,def_pct_stop=0.1, focus_etf_list=None):
    ''' run test for a group of target etf '''
    import pandas as pd
    from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
#    ew_sector_etf_list=['RCD', 'RYH', 'RYT', 'RGI', 'RHS', 'RTM', 'RYF', 'ROOF', 'RSP', 'RYE', 'EQAL', 'EWRE', 'QQEW', 'XBI', 'XAR', 'ROBO','TLT', 'EMLC', 'EEM', 'CURE', 'VXX', 'REM']
    ew_sector_etf_list=['RCD', 'RYH', 'RYT', 'RGI', 'RHS', 'RTM', 'RYF', 'RSP', 'RYE', 'RYU', 'EQAL', 'EWRE', 'QQEW', 'TLT', 'EMLC', 'EEM' ]
#    feat_cols=['TLT', 'EQAL', 'RYF', 'EMLC', 'EWRE' 'RTM', 'RYE', 'REM', 'RYT' 'RYU']

    if focus_etf_list is None:
        focus_etf_list=['TMF']
    all_perf_table_list=[]
    all_trades_df_list=[]
    feat_cols=['TLT', 'EQAL', 'RYF', 'EMLC', 'EWRE', 'RTM', 'RYE', 'EEM', 'RYT', 'RYU', 'RHS']
    for t in focus_etf_list:
        tlist=list(set(ew_sector_etf_list+[t]))
        rank_df=get_rel_nday_ma_zscore_heatmap(tlist, list_tag='rank_sector_etf', use_rank=True, zdays=0)

        all_perf_table_bull, all_trades_df_bull=batch_sector_rotation_learning(rank_df, focus_etf_list=[t], th=th,
                                                    def_pct_stop=def_pct_stop, feat_cols=feat_cols)
        if not all_perf_table_bull is None:
            dlog('all_perf_table_bull:')
            dlog(all_perf_table_bull)
            all_perf_table_list.append(all_perf_table_bull)
            all_trades_df_list.append(all_trades_df_bull)
    if len(all_perf_table_list)>0:
        all_perf_table=pd.concat(all_perf_table_list)
        return all_perf_table, pd.concat(all_trades_df_list)
    return None, None

def test(tlist=['TNA', 'UCO', 'FAS']):
    ''' testing workflow end to end'''
    #bull_lev_etf_list=['SPXL', 'UDOW', 'BNKU', 'UPRO', 'UMDD', 'TMF', 'DRN', 'ERX', 'TNA', 'UCO', 'TECL', 'FAS', 'SOXL', 'GUSH', 'BOIL', 'DUSL']
    #bull_lev_etf_list=['TNA', 'UCO', 'FAS']
    all_perf_table_bull, all_trades_df_bull=run_test(th=0.10,def_pct_stop=0.1, focus_etf_list=tlist)
    return review_perf(all_perf_table_bull), all_trades_df_bull
#test()
