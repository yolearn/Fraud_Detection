import hyperopt
#import lightgbm as lgb

'''
Lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    #'metric': 'auc',
    #'boost_from_average' : False

}
'''
Lgb_params = {
        'num_leaves': 31,
        'min_child_samples': 79,
        'objective': 'binary',
        'max_depth': -1,
        'learning_rate': 0.025,
        "boosting_type": "gbdt",
        #"subsample": 0.8,
        "bagging_seed": 11,
        #"verbosity": -1,
        #'reg_alpha': 0.3,
        #'reg_lambda': 0.3,
        #'colsample_bytree': 0.8,
        'num_boost_round' : 100, 
        'verbose_eval' : 100,
        'early_stopping_rounds' : 50,
        #'scale_pos_weight' : 700,
        'num_threads' :1
        }

#class Hyper_params_tuning():
#   def __init__(self, parameter):        

