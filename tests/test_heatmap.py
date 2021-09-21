import pytest
import unittest
import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def log(msg):
    logger.info(msg)
    
class TestClass(unittest.TestCase):
    def test_rank_heatmap(self):
        import os
        import sys
        from os.path import dirname
        sys.path.append('..')
#        os.chdir('..')
        from datalib.heatmapUtil import get_rel_nday_ma_zscore_heatmap
        tlist=['TLT', 'IWM', 'XLE', 'XLB', 'XLU']
        rank_df=get_rel_nday_ma_zscore_heatmap(tlist, list_tag='rank_sector_etf', use_rank=True, zdays=0)
        log((rank_df.iloc[-1:].values).sum())
        rows, cols=rank_df.shape
        assert(cols==len(tlist))
        assert(rows>1000)
