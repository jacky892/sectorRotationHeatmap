import pytest
import unittest
import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def dlog(msg):
    logger.info(msg)

import os,sys
if not os.path.exists('workflow'):
    sys.path.append('..')
    os.chdir('..')


class TestClass(unittest.TestCase):
    def test_catboost_advisor(self):
        from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
        from workflow.sectorRotationTraderWorkflow import batch_sector_rotation_learning
        from backtest.chandelierExitBacktester import chandelierExitBacktester, backtest_between, get_long_max_drawdown_details
        from datalib.commonUtil import commonUtil as cu
        import pandas as pd
        import pandas_ta as ta
        ew_sector_etf_list=['RCD', 'RYH', 'RYT', 'RGI', 'RHS', 'RTM', 'RYF', 'RSP', 'RYE', 'RYU', 'EQAL', 'EWRE', 'QQEW', 'TLT', 'EMLC', 'EEM' ]
        feat_cols=['TLT', 'EQAL', 'RYF', 'EMLC', 'EWRE', 'RTM', 'RYE', 'EEM', 'RYT', 'RYU', 'RHS']  
        tlist=ew_sector_etf_list.copy()
        tlist.append('TMF')
        rank_df=get_rel_nday_ma_zscore_heatmap(tlist, list_tag='rank_sector_etf', use_rank=True, zdays=0, pred_date='20210901')
        th=0.1
        def_pct_stop=0.1
        focus_etf_tlist=['TMF', 'UCO']
        all_perf_table_bull, all_trades_df_bull=batch_sector_rotation_learning(rank_df, focus_tlist=focus_etf_tlist, th=th,  def_pct_stop=def_pct_stop, feat_cols=feat_cols)

        all_perf_table_bull.to_csv('results/bull_perf.csv')
        all_trades_df_bull.to_csv('results/bull_trades.csv')
        dlog(all_perf_table_bull)
        dlog(all_trades_df_bull)
        assert(len(all_perf_table_bull)>0)
        #    display(mdd_dict)

