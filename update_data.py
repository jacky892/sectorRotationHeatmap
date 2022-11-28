from workflow.sectorRotationTraderWorkflow import update_data
import os,sys
if not os.path.exists('data'):
    os.makedirs('data')

update_data('us_etf')
update_data('fx_tlist')
