''' this module supply standard arguments to call to catboostTradeAdvisor
     so user just need to provide high level like thresholds move (e.g. 10% in 7 days)
    default stop loss def_stop_pct and focus_tlist to get training results '''
from datalib.catboostTradeAdvisor import catboostTradeAdvisor
from backtest.chandelierExitBacktester import dlog
from datalib.commonUtil import commonUtil as cu

def get_cmc_crypto_tlist()->tuple:
    tlist=['BTC', 'ETH', 'BNB', 'XRP', 'DOGE', 'BUSD', 'USDT', 'ADA', 'MATIC', 'DOT', 'LTC', 'SHIB', 'SOL']
    fx_tlist=[f'{c}=C' for c in tlist]
    feat_cols=fx_tlist[:5]
    return fx_tlist, feat_cols

def get_dukas_fx_ticker_list()->tuple:
    fx_tlist=['eurcad', 'gbpusd', 'nzdusd', 'audusd', 'audnzd', 'usdjpy', 'usdchf', 'usdcad',  'usdsgd', 'eurchf', 'eurusd', 'xauusd', 'eurgbp']
    #feat_cols=['eurcad', 'audnzd', 'eurgbp', 'xauusd']
    feat_cols=['eurcad', 'audnzd', 'eurgbp', 'usdjpy', 'usdchf', 'usdcad', 'eurusd', 'xauusd']
    return fx_tlist, feat_cols

def get_lev_etf_list()->tuple:
    import random 
    bull_lev_etf_list=['SPXL', 'UDOW', 'BNKU', 'UPRO', 'UMDD', 'TMF', 'DRN', 'ERX', 'TNA', 'UCO', 'TECL', 'FAS', 'SOXL', 'YCL', 'ULE']
    bear_lev_etf_list=['SMDD', 'TECS', 'SDOW', 'EDZ', 'FAZ', 'BNKD', 'SOXS', 'TZA']
    lev_etf=list(set(bull_lev_etf_list+bear_lev_etf_list))
    return lev_etf, random.sample(lev_etf, 8)

def get_fx_ticker_list()->tuple:
    fx_tlist=['EURUSD=X', 'GBPUSD=X', 'NZDUSD=X', 'AUDUSD=X', 'AUDNZD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X',  'USDSGD=x']
    feat_cols=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCHF=X']
    return fx_tlist, feat_cols

def get_us_etf_list()->tuple:
    #    ew_sector_etf_list=['RCD', 'RYH', 'RYT', 'RGI', 'RHS', 'RTM', 'RYF', 'ROOF', 'RSP', 'RYE', 'EQAL', 'EWRE', 'QQEW', 'XBI', 'XAR', 'ROBO','TLT', 'EMLC', 'EEM', 'CURE', 'VXX', 'REM']
    ew_sector_etf_list=['RCD', 'RYH', 'RYT', 'RGI', 'RHS', 'RTM', 'RYF', 'RSP', 'RYE', 'RYU', 'EQAL', 'EWRE', 'QQEW', 'TLT', 'EMLC', 'EEM', 'SPY']
    feat_cols=['TLT', 'EQAL', 'RYF', 'EMLC', 'EWRE', 'RTM', 'RYE', 'EEM', 'RYT', 'RYU', 'RHS' ]
    return ew_sector_etf_list, feat_cols

def get_jp_etf_list()->tuple:
    '''
    return etf_list and feat_tickers
    1634 Daiwa ETF・TOPIX-17 FOODS
    1635 Daiwa ETF・TOPIX-17 ENERGY RESOURCES
    1636 Daiwa ETF・TOPIX-17 CONSTRUCTION & MATERIALS
    1637 Daiwa ETF・TOPIX-17 RAW MATERIALS & CHEMICALS
    1638 Daiwa ETF・TOPIX-17 PHARMACEUTICAL
    1639 Daiwa ETF・TOPIX-17 AUTOMOBILES & TRANSPORTATION EQUIPMENT
    1640 Daiwa ETF・TOPIX-17 STEEL & NONFERROUS METALS
    1641 Daiwa ETF・TOPIX-17 MACHINERY
    1642 Daiwa ETF・TOPIX-17 ELECTRIC APPLIANCES & PRECISION INSTRUMENTS
    1643 Daiwa ETF・TOPIX-17 IT & SERVICES, OTHERS
    1644 Daiwa ETF・TOPIX-17 ELECTRIC POWER & GAS
    1645 Daiwa ETF・TOPIX-17 TRANSPORTATION & LOGISTICS
    1646 Daiwa ETF・TOPIX-17 COMMERCIAL & WHOLESALE TRADE
    1647 Daiwa ETF・TOPIX-17 RETAIL TRADE
    1648 Daiwa ETF・TOPIX-17 BANKS
    1649 Daiwa ETF・TOPIX-17 FINANCIALS（EX BANKS
    1650 Daiwa ETF・TOPIX-17 REAL ESTATE
    1314 Listed Index Fund S&P Japan Emerging Equity 100
    1316 Listed Index Fund TOPIX100 Japan Large Cap Equity
    1317 Listed Index Fund TOPIX Mid400 Japan Mid Cap Equity
    1318 Listed Index Fund TOPIX Small Japan Small Cap Equity
    1319 Nikkei 300
    1322 Listed Index Fund China A Share (Panda) CSI300
    '''
    etf_list=[ '1314.t', '1316.t', '1317.t', '1318.t', '1319.t', '1322.t',
            '1634.t', '1635.t', '1636.t', '1637.t', '1639.t', '1640.t', '1641.t', '1642.t', 
                '1643.t', '1644.t', '1645.t', '1646.t', '1647.t', '1648.t', '1649.t', '1650.t', 'TLT']
    feat_cols=['TLT', '1635.t', '1638.t', '1639.t', '1644.t', '1645.t', '1648.t', '1650.t', '1316.t' , '1314.t', '1317t.', '1322.t']

    return etf_list, feat_cols

def batch_sector_rotation_learning(rank_df, focus_tlist=['CURE', 'TECL'], th=0.08, retrace_atr_multiple=2,
            def_pct_stop=0.1, feat_cols=['TLT', 'RYE', 'RTM', 'EQAL'], rticker='SPY', exclude_target_col=False ):
    """
    using a time-series rank dataframe to predict movement of tickers in focus_tlist
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
    all_pred_list=[]
    rpdf=cu.read_quote(rticker)
    dlog('latest rpdf: %s ' %  rpdf.iloc[-1])

    for t in focus_tlist:
        if t in feat_cols and exclude_target_col:
            continue
        if not t in rank_df.columns:
            dlog(f'ticker {t} not in rank_df.columns:{rank_df.columns}, skip')
            continue
        dlog (f'now in test getting in batch sector {t}')
        perf_df, trades_df, pred_df=catboostTradeAdvisor.gen_ticker_rank_catboost_results(rank_df, retrace_atr_multiple=retrace_atr_multiple,
                focus_ticker=t, th=th, ex_atr_bars=20, feat_cols=feat_cols, def_pct_stop=def_pct_stop, rticker=rticker)
        if perf_df is None:
            return None, None, None
            
        dlog(perf_df)
        if not perf_df is None:
            table_list.append(perf_df.T)
        if not trades_df is None:
            all_trades_list.append(trades_df)
        if not pred_df is None:
            all_pred_list.append(pred_df)
    if len(table_list)>0:
        all_tb=pd.concat(table_list)

        all_trades_df=pd.concat(all_trades_list)
        all_pred_df=pd.concat(all_pred_list)
        return all_tb, all_trades_df, all_pred_df
    else:
        return None, None, None

def review_perf(all_perf_table):
    '''
    present performance table without the details / debug field
    '''
    all_perf_table['signame']=all_perf_table.index
    all_perf_table['signame']=all_perf_table['signame'].apply(lambda x:x.split('.')[-2])
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
    bull_lev_etf_list=['SPXL', 'UDOW', 'BNKU', 'UPRO', 'UMDD', 'TMF', 'DRN', 'ERX', 'TNA', 'UCO', 'TECL', 'FAS', 'SOXL', 'YCL']
    return bull_lev_etf_list

def run_bull_etf_test(th=0.1, focus_tlist=None):
    '''
    th is the threshold to be passed to catboostTradeAdvisor for labelling
    '''
    import pandas as pd
    if focus_tlist is None:
        focus_tlist=get_bull_etf_tickers()
    all_perf_table, all_trades_df= run_test(th=th, focus_tlist=focus_tlist)
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

def run_bear_etf_test(th=0.1, focus_tlist=None):
    '''
    th is the threshold to be passed to catboostTradeAdvisor for labelling
    '''
    import pandas as pd
    if focus_tlist is None:
        focus_tlist=get_bear_etf_tickers()
    all_perf_table, all_trades_df= run_test(th=th, focus_tlist=focus_tlist)
    if not all_perf_table is None:
        review_perf(all_perf_table)
    return all_perf_table, all_trades_df

def run_forex_worflow(pred_date=None, use_dukas=False, focus_tlist=None, heatmap_fname='dfx_hm.jpg'):
    from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
    from datalib.commonUtil import commonUtil as cu   

    if use_dukas:
        fx_tlist, feat_cols= get_dukas_fx_ticker_list()
        if focus_tlist is None:
            focus_tlist=[ 'eurusd', 'usdjpy']
        
        skip_cnt=6
        rticker='xauusd'
        th=0.01
        def_pct_stop=0.02
    else:
        fx_tlist, feat_cols= get_fx_ticker_list()
        if focus_tlist is None:
           focus_tlist=['YCL', 'ULE']
        rticker='UUP'    
        skip_cnt=5
        cu.download_quote(rticker)
        th=0.03
        def_pct_stop=0.05
        
    from workflow.sectorRotationTraderWorkflow import update_data
    #update_data('fx_tlist')
    from datalib.commonUtil import commonUtil as cu
    
    rank_df=get_rel_nday_ma_zscore_heatmap(fx_tlist+focus_tlist, list_tag='rank_fx', use_rank=True, zdays=0, pred_date=pred_date, skip_cnt=skip_cnt)
    feat_cols=fx_tlist
    dlog(rank_df.tail())
    all_perf_table, all_trades_df, pred_df=batch_sector_rotation_learning(rank_df, focus_tlist=focus_tlist, th=th, rticker=rticker,
                                                              def_pct_stop=def_pct_stop, feat_cols=feat_cols)
    return  all_perf_table, all_trades_df, pred_df 

def run_cmc_crypto_worflow(pred_date='20221020', update_data=False):
    from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
    from workflow.sectorRotationTraderWorkflow import  get_fx_ticker_list, batch_sector_rotation_learning
    from backtest.chandelierExitBacktester import chandelierExitBacktester, backtest_between, get_long_max_drawdown_details
    from datalib.commonUtil import commonUtil as cu
    import pandas as pd
    import pandas_ta as ta
    
    import random

    feat_cols=list(set(['BTC=C', 'ETH=C'] + list(random.sample(fx_tlist, 3))))
    feat_cols, fx_tlist
    if update_data:
        for t in fx_tlist:
            cu.download_quote(t)

    focus_etf_tlist=['ETH=C']
    from workflow.sectorRotationTraderWorkflow import update_data
    from datalib.commonUtil import commonUtil as cu
    rticker='USDT=C'
    cu.download_quote(rticker)
    rank_df=get_rel_nday_ma_zscore_heatmap(fx_tlist+focus_etf_tlist, list_tag='rank_fx', use_rank=True, zdays=0, pred_date=pred_date)
    th=0.1
    def_pct_stop=0.05
    feat_cols=fx_tlist
    all_perf_table, all_trades_df, pred_df=batch_sector_rotation_learning(rank_df, focus_tlist=focus_etf_tlist, th=th, rticker=rticker,
                                                              def_pct_stop=def_pct_stop, feat_cols=feat_cols)
    ret_dict={}

    return  all_perf_table, all_trades_df, pred_df


def run_etf_test(th=0.10,def_pct_stop=0.1, mkt='usa', focus_tlist=['TMF'], rticker='SPY', heatmap_fname='yf_hm.jpg'):
    ''' run test for a group of target etf '''
    import pandas as pd
    from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
    if mkt=='japan':
        sector_etf_list, feat_cols=get_jp_etf_list()
    else:
        sector_etf_list, feat_cols=get_us_etf_list()
    all_perf_table_list=[]
    all_trades_df_list=[]
    for t in focus_tlist:
        tlist=list(set(sector_etf_list+[t]))
        rank_df=get_rel_nday_ma_zscore_heatmap(tlist, list_tag='rank_sector_etf', use_rank=True, zdays=0)

        all_perf_table_bull, all_trades_df_bull, pred_df=batch_sector_rotation_learning(rank_df, focus_tlist=[t], th=th,
                                                    def_pct_stop=def_pct_stop, feat_cols=feat_cols, rticker=rticker)
        if not all_perf_table_bull is None:
            dlog('all_perf_table_bull:')
            dlog(all_perf_table_bull)
            all_perf_table_list.append(all_perf_table_bull)
            all_trades_df_list.append(all_trades_df_bull)
    if len(all_perf_table_list)>0:
        all_perf_table=pd.concat(all_perf_table_list)
        return all_perf_table, pd.concat(all_trades_df_list)
    return None, None, None

def extract_pred_table_for_sector_rotation(all_pred_df, ticker):
    import pandas as pd
    all_pred_df['date']=all_pred_df.index
    if 'y' in all_pred_df.columns:
        all_pred_df.drop('y', axis=1, inplace=True)
    td=pd.to_timedelta('5 days')
    ts=pd.to_datetime('now')
    ts2=ts-td
    bidx=all_pred_df['date']>ts2
    _send_df=all_pred_df[bidx].query('pred_y>0')
    if len(_send_df)==0:
        return None, f'no suggestion for ticker {ticker}'
    import dataframe_image as dfi
    ofname=f'{ticker}_pred_df.jpg'
    dfi.export(_send_df, ofname, table_conversion='matplotlib')
    return _send_df, ofname


def run_chatbot_sector_pred_for_ticker(focus_ticker, peer='etf'):
    from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
    from workflow.sectorRotationTraderWorkflow import batch_sector_rotation_learning
    from backtest.chandelierExitBacktester import chandelierExitBacktester, backtest_between, get_long_max_drawdown_details
    from datalib.commonUtil import commonUtil as cu
    import pandas as pd
    import pandas_ta as ta
    ew_sector_etf_list=['RCD', 'RYH', 'RYT', 'RGI', 'RHS', 'RTM', 'RYF', 'RSP', 'RYE', 'RYU', 'EQAL', 'EWRE', 'QQEW', 'TLT', 'EMLC', 'EEM' ]
    feat_cols=['TLT', 'EQAL', 'RYF', 'EMLC', 'EWRE', 'RTM', 'RYE', 'EEM', 'RYT', 'RYU', 'RHS']  

    peer_feat_dict={}
    peer_hm_dict={}
    rticker_dict={}
    th_dict={}
    th_dict['etf']=0.05
    th_dict['cmc']=0.08    
    th_dict['fx']=0.012        
    rticker_dict['etf']='IWV'
    rticker_dict['cmc']='BCH=C'    
#    rticker_dict['fx']='USDCHF=X'        
    rticker_dict['fx']='usdchf'        
    peer_hm_dict['etf'],peer_feat_dict['etf']=get_us_etf_list()
    peer_hm_dict['fx'],peer_feat_dict['fx']=get_dukas_fx_ticker_list()
    peer_hm_dict['cmc'],peer_feat_dict['cmc']=get_cmc_crypto_tlist()
    rticker=rticker_dict[peer]
    
    tlist=peer_hm_dict[peer]
    tlist.append(focus_ticker)
    feat_cols=peer_feat_dict[peer]
    rticker=rticker_dict[peer]

    th=th_dict[peer]
    def_pct_stop=th
    skip_cnt=6
    cu.download_quote(rticker)
    feat_cols.append(focus_ticker)
    feat_cols=list(set(feat_cols))
    focus_tlist=[focus_ticker]
    for t in list(set(feat_cols+tlist+[rticker])):
        print('XxXXXXXXXXXXXXXXX downloading data ', t)
        cu.download_quote(t)
    
    rank_df=get_rel_nday_ma_zscore_heatmap(tlist, focus=focus_ticker, list_tag=f'rank_{peer}', use_rank=True, zdays=0, pred_date=None, imgofname='xhm.jpg', skip_cnt=skip_cnt) 

    print(rank_df.tail())
    all_perf_table, all_trades_df, all_pred_df=batch_sector_rotation_learning(rank_df, focus_tlist=focus_tlist, th=th, rticker=rticker,
                                                              def_pct_stop=def_pct_stop, feat_cols=feat_cols)
    all_perf_table, all_trades_df, all_pred_df
    return all_perf_table, all_trades_df, all_pred_df

class sectorRotationTraderWorkflow:
    @staticmethod
    def run_chatbot_sector_pred_for_ticker(focus_ticker, peer='etf'):
        return  run_chatbot_sector_pred_for_ticker(focus_ticker, peer)

    @staticmethod
    def extract_pred_table_for_sector_rotation(all_pred_df, ticker):
        return extract_pred_table_for_sector_rotation(all_pred_df, ticker)


def test(tlist=['TNA', 'UCO', 'FAS']):
    ''' testing workflow end to end'''
    #bull_lev_etf_list=['SPXL', 'UDOW', 'BNKU', 'UPRO', 'UMDD', 'TMF', 'DRN', 'ERX', 'TNA', 'UCO', 'TECL', 'FAS', 'SOXL', 'GUSH', 'BOIL', 'DUSL']
    #bull_lev_etf_list=['TNA', 'UCO', 'FAS']
    all_perf_table_bull, all_trades_df_bull=run_test(th=0.10,def_pct_stop=0.1, focus__list=tlist)
    return review_perf(all_perf_table_bull), all_trades_df_bull
#test()
def update_data(ticker_group='us_etf'):
    tlist_dict={}
    tlist_dict['us_etf']=get_us_etf_list()
    tlist_dict['fx_tlist']=get_fx_ticker_list()
    tlist_dict['dukas_fx_tlist']=get_dukas_fx_ticker_list()
    tlist_dict['cmc_crypto']=get_dukas_fx_ticker_list()
    tlist_dict['lev_etf']=get_lev_etf_list()
    tlist, feat_cols=tlist_dict[ticker_group]
    full_list=list(tlist)+list(feat_cols)
    for ticker in full_list:
        cu.download_quote(ticker)



