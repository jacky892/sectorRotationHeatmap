from collections import defaultdict
import pandas as pd
from backtest.chandelierExitBacktester import chandelierExitBacktester as backtester
from datalib.commonUtil import commonUtil as cu
from backtest.chandelierExitBacktester import dlog


def split_data_classifier(feat_df, feat_cols, target_col='_n_ret', th=-0.05, test_sample_cnt=200):
    '''
    if th is array, e.g. [-0.05, 0.05], it will return 3 ea
    if target_col start with _n, e.g. _n_ret15, then th will be used to create flag
    if target_col start with _lb, it will be used as is
    Keyword arguments:
    :DataFrame feat_df: return from heatmapUtil.py get_rel_nday_ma_zscore_heatmap
    returns dict
        Where keys = ['X_train', 'y_train', 'X_test', 'y_test']
    '''
    dlog(f'target_col:,{target_col}, {feat_df[feat_cols].tail(2)}')
    _=feat_df.ffill().copy()
    if target_col[:3]=='_n_':
        if type(th)==type(0.1):
            if th>0:
                y=(_[target_col]>th)*1.0
            else:
                y=(_[target_col]<th)*1.0
#        if type(th)==type([]): ##
        if isinstance(th, list ): ##
            y=(_[target_col]>th[-1])*1.0
            bidx=_[target_col]<th[0]
            idx=y[bidx].index
            y.loc[idx]=-1
    else:
        y=feat_df[target_col]
    X=_[feat_cols].ffill()
    if test_sample_cnt<1:
        test_sample_cnt=int(len(feat_df)*test_sample_cnt)
    X_train=X.iloc[:-test_sample_cnt]
    X_test=X.iloc[-test_sample_cnt:]
    y_train=y.iloc[:-test_sample_cnt]
    y_test=y.iloc[-test_sample_cnt:]
    ret_dict=defaultdict()
    ret_dict['X_train']=X_train
    ret_dict['X_test']=X_test
    ret_dict['y_train']=y_train
    ret_dict['y_test']=y_test
    ret_dict['full_table']=feat_df

    return ret_dict

def simple_xgboost_learner(ticker, feats_df, feat_cols, target_col, th=-0.05, test_sample_cnt=300, longshort='long', show_cv=False, param={}):
    import pandas as pd
    import numpy as np
    import xgboost
    from sklearn import metrics
    dlog(f'{longshort} {ticker} at cat boost:{feats_df.tail(2)}')
    split_dict=split_data_classifier(feats_df,  feat_cols, target_col, th=th, test_sample_cnt=test_sample_cnt)

    train_x=split_dict['X_train']
    train_y=split_dict['y_train']
    test_x=split_dict['X_test']
    test_y=split_dict['y_test']
    dlog('test_y desc',test_y.describe())
#    dlog('train_y desc',train_y.describe())

    import xgboost
    missing=float('nan')
    from xgboost import XGBClassifier
    n_estimators=100
    max_depth=5 
    model = XGBClassifier(learning_rate=0.001, n_estimators=n_estimators, max_depth=max_depth, scale_pos_weight=6, objective='binary:logistic', eval_metric='error')

    model.fit(train_x, train_y)

    plst = list(param.items())+[('eval_metric', 'ams@0.15')]

    num_round = 200
    dlog('loading data end, start to boost trees')
    dlog('model:', model, model.get_booster().get_fscore())
    unique_y=list(set(train_y))
    dlog(f'unique_y: {unique_y, train_y.shape, test_y.shape, test_x.shape}')
    model_best_iteration=model.get_booster().best_iteration
    y_pred = model.predict(test_x)
    
    dlog('pred 1 is %s %s ' % ( y_pred[-10:], len(y_pred)))
    pred_flag = [round(value) for value in y_pred]
    dlog('pred fixed is %s %s ' % (pred_flag[-10:], len(pred_flag)))
    score1=metrics.accuracy_score(test_y, pred_flag)
    score2=metrics.confusion_matrix(test_y, pred_flag)
    score3=metrics.classification_report(test_y, pred_flag)
    iclasses=sorted(list(set(train_y)))
    iclasses_t=sorted(list(set(list(np.unique(test_y))+list(np.unique(pred_flag)))))
    if iclasses[0]==-1:
        classes_dict_nways={-1:'drop', 0:'sideway', 1:'rise', 2:'rise'}
    else:
        classes_dict_nways={-1:'drop', 0:'drop', 1:'sideway', 2:'rise', 3:'rise'}
    test_classes_name= [classes_dict_nways[c] for c in iclasses_t]
    if len(test_classes_name)==1:
        test_classes_name=['True', 'False']
    dlog(f'test_classes_names: names, dict, ilist {test_classes_name, classes_dict_nways, iclasses, iclasses_t}')
    cfm_fname='tmp/a.jpg'
#    dlog(f'scores {score1, feat_cols}')
    dlog(f'score2:{score2}')
    dlog(f'report {score3}'.replace('report', 're') )
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support= precision_recall_fscore_support(test_y, pred_flag)
    #dlog(f'precision:{precision}, {recall}, {f1}, {best_iteration}, {support} ')
    rel_cols=feat_cols
    feature_importances=model.feature_importances_
    rf_importances = pd.DataFrame({'name':rel_cols[:len(feature_importances)],
                                        'imp_val':feature_importances
                                          }).sort_values(by='imp_val',
                                           ascending=False).reset_index(drop=True)
    dlog(rf_importances.head(5))
    dlog(' pred_flag:',)
    ret_df=pd.DataFrame()
    ret_df['y']=test_y
    ret_df['pred_y']=pred_flag
#    dlog(ret_df)
#    dlog(pred_flag.tail(5))
    ret_dict=defaultdict()
    ret_dict['rf_importances']=rf_importances
    ret_dict['score1']=score1
    ret_dict['score2']=score2
    ret_dict['score3']=score3
    ret_dict['recall']=recall
    ret_dict['cfm_fname']=cfm_fname
    ret_dict['f1']=f1
    ret_dict['pred_df']=ret_df
    ret_dict['model']=model
    ret_dict['model_best_iteration']=model_best_iteration
    ret_dict['split_dict']=split_dict

    return ret_dict



def simple_catboost_learner(ticker, feats_df, feat_cols, target_col, th=-0.05, test_sample_cnt=300, longshort='long', show_cv=False, param={}):
    '''
    Keyword arguments:
    :DataFrame feats_df : dataframes with all the features in feat_cols
    :list feat_cols: list of columns to be used as features, e.g. ['TLT', 'RYT', 'EQAL']
    :str target_col: added by e.g. backtester.add_labels_by_bars_lb_7_spikeup
    :DataFrame rank_df: return from heatmapUtil.py get_rel_nday_ma_zscore_heatmap
    :str focus_ticker: ticker's trading signal you want to train for
    :returns: dict
        WHERE
        dict include 'rf_importances', 'pred_y' and other metrics
    '''

    import pandas as pd
    import numpy as np
    from sklearn import metrics
    dlog(f'{longshort} {ticker} at cat boost:{feats_df.tail()}')
    split_dict=split_data_classifier(feats_df,  feat_cols, target_col, th=th, test_sample_cnt=test_sample_cnt)

    train_x=split_dict['X_train']
    train_y=split_dict['y_train']
    test_x=split_dict['X_test']
    test_y=split_dict['y_test']
    from catboost import Pool, CatBoostClassifier, cv
    loss_function='Logloss'

    custom_metric=['AUC:hints=skip_train~false', loss_function]
    if type(th)==type([]):
        loss_function='MultiClass'
        custom_metric=[loss_function]

    cate_features_index = np.where(train_x[feat_cols].dtypes != float)[0]
    metric_period=150
    if param==defaultdict():
        param['task_type']='CPU'
#        param['task_type']='GPU'
        param['depth']=7
        param['l2_leaf_reg']=1e-19
    depth=param['depth']
    l2_leaf_reg = param['l2_leaf_reg']
    import socket
    hostname=socket.gethostname()
    tdir=f'catinfo.{hostname}'
    task_type=param['task_type']
    cat_model = CatBoostClassifier(iterations=1800, loss_function=loss_function,
                            border_count=128,
                            grow_policy='Lossguide',
                            max_leaves=48,
#                            od_type,'IncToDec',
                               train_dir=tdir, task_type=task_type,
                               custom_metric=custom_metric,
                               use_best_model=True,
                               max_depth=depth, l2_leaf_reg=l2_leaf_reg,
                               learning_rate=0.002,
                                auto_class_weights='Balanced',
                               bootstrap_type='Bayesian',
    #                               bootstrap_type='Poisson',
                               verbose=300,
                               random_seed=42, metric_period=metric_period)
    model=cat_model
    unique_y=list(set(train_y))
    dlog(f'unique_y: {unique_y, train_y.shape, test_y.shape, test_x.shape}')
    if len(unique_y)==1:
        return None
    if len(cate_features_index)==0:
        model.fit(train_x, train_y,eval_set=(test_x, test_y))
    else:
        model.fit(train_x,train_y, eval_set=(test_x,test_y))
    best_iteration=model.get_best_iteration()
    model_best_iteration=model.best_iteration_

    if show_cv:
        #for stock application don't include test_x in cross validation
#        cv_data = cv(pool=Pool(pd.concat([train_x, test_x]),list(train_y)+list(test_y)), params=model.get_params(), fold_count=3, type='TimeSeries', plot=True, as_pandas=True)
        cv_data = cv(pool=Pool(pd.concat([train_x]),list(train_y)), params=model.get_params(),
                        fold_count=4, type='TimeSeries', plot=True, as_pandas=True)
        dlog(f'model:{model} cv_data{cv_data}')

    pred_flag=model.predict(test_x)
    score1=metrics.accuracy_score(test_y, pred_flag)
    score2=metrics.confusion_matrix(test_y, pred_flag)
    score3=metrics.classification_report(test_y, pred_flag)
    iclasses=sorted(list(set(train_y)))
    iclasses_t=sorted(list(set(list(np.unique(test_y))+list(np.unique(pred_flag)))))
    if iclasses[0]==-1:
        classes_dict_nways={-1:'drop', 0:'sideway', 1:'rise', 2:'rise'}
    else:
        classes_dict_nways={-1:'drop', 0:'drop', 1:'sideway', 2:'rise', 3:'rise'}
    test_classes_name= [classes_dict_nways[c] for c in iclasses_t]
    if len(test_classes_name)==1:
        test_classes_name=['True', 'False']
    dlog(f'test_classes_names: names, dict, ilist {test_classes_name, classes_dict_nways, iclasses, iclasses_t}')
    cfm_fname='tmp/a.jpg'
#    dlog(f'scores {score1, feat_cols}')
    dlog(f'score2:{score2}')
    dlog(f'report {score3}'.replace('report', 're') )
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support= precision_recall_fscore_support(test_y, pred_flag)
    dlog(f'precision:{precision}, {recall}, {f1}, {best_iteration}, {support} ')
    rel_cols=feat_cols
    rf_importances = pd.DataFrame({'name':rel_cols[:len(model.feature_importances_)],
                                        'imp_val':model.feature_importances_
                                          }).sort_values(by='imp_val',
                                           ascending=False).reset_index(drop=True)
    dlog(rf_importances.head(5))
    dlog(' pred_flag:',)
    ret_df=pd.DataFrame()
    ret_df['y']=test_y
    ret_df['pred_y']=pred_flag
    dlog(ret_df.tail())
    ret_dict=defaultdict()
    ret_dict['rf_importances']=rf_importances
    ret_dict['score1']=score1
    ret_dict['score2']=score2
    ret_dict['score3']=score3
    ret_dict['recall']=recall
    ret_dict['cfm_fname']=cfm_fname
    ret_dict['f1']=f1
    ret_dict['pred_df']=ret_df
    ret_dict['model']=model
    ret_dict['model_best_iteration']=model_best_iteration
    ret_dict['split_dict']=split_dict

    return ret_dict

def check_catboost_installed():
    ret=True
    try:
        import catboost
        print("module 'catboost' is installed")
    except Exception as e:
        print('error', e)
        ret=False
    return ret

class catboostTradeAdvisor:
    @staticmethod
    def learn_from_sector_matrix_df(rank_df, focus_ticker='QQQ', feat_cols=['TLT', 'XLV'], pred_bars=7,
            threshold=0.1, retrace_atr_multiple=3, ex_atr_bars=20, def_pct_stop=0.1, test_sample_cnt=80, rticker='SPY'):
        '''
        wrapper function called by gen_ticker_rank_catboost_results for add labels and then call simple_catboost_learner
        Keyword arguments:
            :DataFrame rank_df: return from heatmapUtil.py get_rel_nday_ma_zscore_heatmap
            :str focus_ticker: the ticker whose return will be used for labeling, say > threshold movement in 7 days
            :int retrace_atr_multiple: the number of atr retracement before we exit a position
        Returns: all_bt_dict
            where all_bt_dict is a dict for backtest results
            e.g. keys:
        '''
        maxrank=rank_df.max().max()
        rxdf=rank_df/maxrank
        ret_df=pd.DataFrame()
        if focus_ticker not in feat_cols:
            ret_df[f'i_{focus_ticker}']=rxdf[focus_ticker]

        for t in feat_cols:
            if t in rxdf.columns:
                ret_df[f'i_{t}']=rxdf[t]
            ret_df[f'i_{t}_d']=rxdf[t].diff(1)
            ret_df[f'i_{t}_d2']=rxdf[t].diff(1).diff(1)

        pdf=cu.read_quote(focus_ticker)
        bars=pred_bars
        label_cols=backtester.add_labels_by_bars(pdf, bars=bars, threshold=threshold)
        if label_cols is None:
            return "insufficent data"

#        dlog(f'possible target cols: {label_cols}')
        ndf=ret_df.merge(pdf, left_index=True, right_index=True)
        dlog(f'label cols:{label_cols}')
        feat_cols=[col for col in ndf.columns if col[:2]=='i_']
        dlog(f'adjusted feat cols: {feat_cols}')
        all_ret_dict=defaultdict()


    #    target_cols=[f'_lbm_{day}_bigmove', f'_lbm_{day}_bigspike'    ]
    #    for target_col in target_cols:
    #        x=simple_catboost_learner(focus_ticker, ndf, feat_cols, target_col, th=[-threshold, threshold], test_sample_cnt=test_sample_cnt, param={})
    #        dlog(f'bt dict keys:{x.keys()}')
    #        all_ret_dict[target_col]=x
        target_cols=[  f'_lb_{bars}_spikeup' , f'_lb_{bars}_bigrise' , f'_lb_{bars}_bigdrop' ]
        use_catboost=check_catboost_installed()
        use_catboost=False
        dlog(f'before bt rticker:{rticker}')
        for target_col in target_cols:
            if not use_catboost:
                x=simple_xgboost_learner(focus_ticker, ndf, feat_cols, target_col, th=threshold, test_sample_cnt=test_sample_cnt, param={})
            else:        
                x=simple_catboost_learner(focus_ticker, ndf, feat_cols, target_col, th=threshold, test_sample_cnt=test_sample_cnt, param={})
            all_ret_dict[target_col]=x

    #    target_cols=[ f'_lb_{day}_spikedown',  f'_lb_{day}_bigdrop' ]
    #    for target_col in target_cols:
    #        x=simple_catboost_learner(focus_ticker, ndf, feat_cols, target_col, th=threshold, test_sample_cnt=test_sample_cnt, param={})
    #        all_ret_dict[target_col]=x

        img_list=[]
        ret_df_list=[]
        all_bt_dict=defaultdict()
        for lb_key, ret_dict in all_ret_dict.items():
            if ret_dict is None:
                continue
            ret_df=ret_dict['pred_df']
            pred_df1=ret_df
            cfmatrix=ret_dict['cfm_fname']
            rf_importances=pd.DataFrame()
            if 'rf_importances' in ret_dict:
                rf_importances=ret_dict['rf_importances']
            model_best_iteration=-1
            img_list.append(cfmatrix)
            ret_df_list.append(ret_df)
            longshort='long'
            if 'down' in lb_key or 'drop' in lb_key:
                longshort='short'
            signame=f'catboost.{lb_key}.{longshort}'
            pred_df1['signame']=signame
            pred_df1['longshort']=longshort
            pred_df1['ticker']=focus_ticker
            bt_dict=backtester.add_backtest_from_pred_df(focus_ticker, pred_df1, pred_col='pred_y', rticker=rticker,
                         trade_type=longshort, signame=signame, retrace_atr_multiple=retrace_atr_multiple, def_pct_stop=def_pct_stop)
            pred_df1.to_csv(f'results/{focus_ticker}.{signame}.pred.csv')
            rf_importances.to_csv(f'results/{focus_ticker}.{signame}.feat_rank.csv')
            if bt_dict is None:
                continue
            if bt_dict['trades_df'] is None:
                continue
            bt_dict['trades_df'].to_csv(f'results/{focus_ticker}.{signame}.trades.csv')
            bt_dict['pred_df']=pred_df1
            bt_dict['signame']=signame
            if 'model_best_iteration' in  ret_dict.keys():
                model_best_iteration=ret_dict['model_best_iteration']
            bt_dict['all_tradesummary']['model_best_iteration']=model_best_iteration
            all_bt_dict[signame]=bt_dict
        return all_bt_dict

    @staticmethod
    def gen_ticker_rank_catboost_results(rank_df, focus_ticker='XLE', th=0.08, retrace_atr_multiple=3, def_pct_stop=0.1,
             ex_atr_bars=20, feat_cols=['TLT', 'XLB', 'VNQ', 'XLC'], rticker='SPY')->tuple:
        '''
        Keyword arguments:
        :DataFrame rank_df: return from heatmapUtil.py get_rel_nday_ma_zscore_heatmap
        :str focus_ticker: ticker's trading signal you want to train for
        :returns: tuple (all_perf_table, all_trades_df)
            WHERE
            DataFrame all_perf_table all ticker-label pairs performance
            DataFrame all_trades_df all trades triggered by the prediction in test set
        '''
        import pandas as pd
        rank_cols=feat_cols.copy()
        rank_cols.append(focus_ticker)
        test_sample_cnt=int(len(rank_df)*0.2)
        ret_dict=catboostTradeAdvisor.learn_from_sector_matrix_df(rank_df, focus_ticker=focus_ticker, feat_cols=rank_cols, threshold=th, retrace_atr_multiple=retrace_atr_multiple, ex_atr_bars=ex_atr_bars, def_pct_stop=def_pct_stop, rticker=rticker, test_sample_cnt=test_sample_cnt)
        if type(ret_dict)==type(''): # return error message
            return None, None, ret_dict            
        dlog(f'{focus_ticker} results:', ret_dict.keys())
        def get_simple_dict_field(big_dict):
            import pandas as pd
            new_dict=defaultdict()
            for k,v in big_dict.items():
                if not type(v) in [type({}), type(pd.DataFrame())]:
                    new_dict[k]=v
            return new_dict

#        pred_df=ret_dict['pred_df']
        all_row_dict=defaultdict()
        all_pred_df_list=[]
        all_pred_df=[]
        all_trades_df_list=[]
        all_perf_table=None
        all_trades_df=None
        for k,v in ret_dict.items():
            if v is None:
                continue
            big_dict=v['all_tradesummary']
            trades_df=v['trades_df']
            pred_df=v['pred_df']
            all_pred_df_list.append(pred_df)
            all_trades_df_list.append(trades_df)
            row_dict=get_simple_dict_field(big_dict)
            all_row_dict[f'{focus_ticker}.{k}']=row_dict
        if len(all_row_dict)>0:
            all_perf_table=pd.DataFrame(all_row_dict)
        if len(all_trades_df_list)>0:
            all_trades_df=pd.concat(all_trades_df_list)
        if len(all_pred_df_list)>0:
            all_pred_df=pd.concat(all_pred_df_list)
        return all_perf_table, all_trades_df, all_pred_df
