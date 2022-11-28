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
    def test_updateData(self):
        from datalib.commonUtil import commonUtil as cu
        tlist=['TLT', 'IWM', 'XLE', 'XLB', 'XLU']
        for ticker in tlist:
            df=cu.download_yf_quote(ticker)
        
            print(df)
            assert(len(df)>1000)
