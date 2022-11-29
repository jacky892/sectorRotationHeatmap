from workflow.sectorRotationTraderWorkflow import test 
def test_dukasfx():
    from workflow.sectorRotationTraderWorkflow import  run_forex_worflow
    all_perf_table, all_trades_df=run_forex_worflow(pred_date='20221126', use_dukas=True, focus_tlist=['eurusd', 'usdjpy', 'audnzd', 'usdcad'])
    print("latest prediction:",all_trades_df)       
    #assert(all_perf_table.exposure_adjusted_annualized_gain.mean()>0)

test_dukasfx()
