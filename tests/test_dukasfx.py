import pytest
import unittest
import logging
logging.basicConfig()
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import os,sys
if not os.path.exists('workflow'):
    sys.path.append('..')
    os.chdir('..')

def log(msg):
    logger.info(msg)
    
class TestClass(unittest.TestCase):
    def test_dukasfx(self):
        from workflow.sectorRotationTraderWorkflow import  run_forex_worflow
        all_perf_table, all_trades_df=run_forex_worflow(pred_date='20221126', use_dukas=True)
                
        assert(len(all_perf_table)>0)
        assert(len(all_trades_df)>0)
        #assert(all_perf_table.exposure_adjusted_annualized_gain.mean()>0)
