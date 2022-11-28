import pytest
import unittest
import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os,sys
sys.path.append('..')
if not os.path.exists('workflow'):
    os.chdir('..')

def log(msg):
    logger.info(msg)

    
class TestClass(unittest.TestCase):
    def test_rank_heatmap(self):
        from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
        tlist=['TLT', 'IWM', 'XLE', 'XLB', 'XLU']
        rank_df=get_rel_nday_ma_zscore_heatmap(tlist, list_tag='rank_sector_etf', use_rank=True, zdays=0, pred_date='20210901')
        log((rank_df.iloc[-1:].values).sum())
        rows, cols=rank_df.shape
        assert(cols==len(tlist))
        assert(rows>1000)
