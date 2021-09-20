import pytest
import unittest
import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def dlog(msg):
    logger.info(msg)

import os
import sys
from os.path import dirname
sys.path.append('..')
    
class TestClass(unittest.TestCase):

    def test_backtest_with_max_drawdown(self):
        from backtest.chandelierExitBacktester import chandelierExitBacktester, backtest_between, get_long_max_drawdown_details
        from backtest.chandelierExitBacktester import commonUtil as cu
        import pandas as pd
        import pandas_ta as ta
        chandelierExitBacktester()
        entry_date=pd.to_datetime('20210517')
        exit_date=pd.to_datetime('20210903')

        rticker='SPY'
        import pandas as pd
        ticker='FAS'
        all_signal_dict={}
    #    all_signal_dict[1]={'entry_date':'20210708', 'ticker':ticker}
    #    all_signal_dict[2]={'entry_date':'20210817', 'ticker':ticker}
        all_signal_dict[2]={'entry_date':'20210719', 'ticker':ticker}

        df=pd.DataFrame(all_signal_dict).T
        price_df=cu.read_quote(ticker)
        rprice_df=cu.read_quote(rticker)

        tradedates_df=chandelierExitBacktester.get_ch_ex_trade_exit_date(df, price_df, ticker=ticker, def_pct_stop=0.1, atr_multiple=3, plot=True)
        dlog(tradedates_df)
        summary_dict, trades_df=chandelierExitBacktester.gen_ch_ex_trades_from_tradedates(tradedates_df, price_df, ticker=ticker)
        trades_df
        for k,v  in trades_df.iloc[:1].iterrows():
            subdf=price_df.loc[v['entry_date']:v['exit_date']]
            mdd_dict=get_long_max_drawdown_details(subdf, plot=True)
        print(mdd_dict.keys())
        print(mdd_dict['mdd'])
        assert('mdd_exceeded' in mdd_dict.keys())
        assert('mdd' in mdd_dict.keys())
        assert(mdd_dict['mdd'] <-0.1)
        ret_ser=mdd_dict['ret_ser']
        bidx=ret_ser>0.05

