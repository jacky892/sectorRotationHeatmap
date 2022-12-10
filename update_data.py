from workflow.sectorRotationTraderWorkflow import update_data
import os,sys
if not os.path.exists('data'):
    os.makedirs('data')

#update_data('us_etf')
#update_data('fx_tlist')
#update_data('dukas_fx_tlist')
update_data('cmc_crypto')
update_data('lev_etf')

