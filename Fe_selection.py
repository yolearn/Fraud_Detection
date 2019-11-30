import time
Start_time = time.time()
from Preprocessing import Processed_data
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from Tuning_parameter import Lgb_params
from sklearn.metrics import f1_score
import numpy as np
import lightgbm as lgb
import pandas as pd
import os 
from datetime import datetime
from sklearn.feature_selection import RFECV
#from Tuning_parameter import Lgb_params
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
Start_time = time.time()

Train_df      = Processed_data['Train_df']
Test_df       = Processed_data['Test_df']
Test_id       = Processed_data['Test_df']['txkey']
#Record_list   = Processed_data['Record_list']
Cate_feat     = Processed_data['Cate_feat']

N_folds       = Processed_data['N_folds']
Target        = 'fraud_ind'
#Time          = datetime.today().strftime('%m-%d-%H-%M')

def Cal_f1_score(True_y, Pre_y):
    return f1_score(True_y, Pre_y)
            
def F1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def Split_group_kfolds():
    Train_X = Train_df.drop(['fraud_ind'], axis=1)
    print(Train_X['Month'].unique())
    Train_Y = Train_df['fraud_ind']
    Folds  = GroupKFold(n_splits=3)
    Splited_data = Folds.split(Train_X, Train_Y, groups=Train_X['Month']) 
    return Splited_data


clf = lgb.LGBMClassifier( 
                #early_stopping_rounds = 10, 
                #categorical_feature = Cate_feat, 
                #feval=F1_score, 
                #evals_result={},     
                num_leaves =  31,
                min_child_samples =  79,
                objective = 'binary',
                max_depth = -1,
                learning_rate =  0.025,
                boosting_type = 'gbdt',
                #"subsample": 0.8,
                bagging_seed = 42,
                #"verbosity": -1,
                #'reg_alpha': 0.3,
                #'reg_lambda': 0.3,
                #'colsample_bytree': 0.8,
                n_estimators = 1, 
                #verbose_eval = 100,
                #'scale_pos_weight' : 700,
                #num_threads  =4
                )

rfecv = RFECV(estimator=clf, step=1, cv = Split_group_kfolds(), verbose=2, n_jobs = -1, scoring = 'f1')

Exluded_cols = ['fraud_ind', 'txkey', 'Month']
All_Features = [Feat for Feat in Train_df.columns if Feat not in Exluded_cols]

rfecv.fit(Train_df[All_Features], Train_df[Target])
sel_features = [f for f, s in zip(All_Features, rfecv.support_) if s]
print('\n The selected features are {}:'.format(sel_features))
print(len(sel_features))

plt.figure(figsize=(12, 9))
plt.xlabel('Number of features ')
plt.ylabel('Cross-validation score (F1_score)')
plt.plot(range(1, len(rfecv.grid_scores_) + 1) , rfecv.grid_scores_)
plt.show()

print((time.time() - Start_time)/60, 'minute')