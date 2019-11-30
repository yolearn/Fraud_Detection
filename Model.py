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
from Tuning_parameter import Lgb_params
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

class BaseModel():
    def __init__(self, processed_data):
        print('Base class initializing...')
        self.Train_df      = processed_data['Train_df']
        self.Test_df       = processed_data['Test_df']
        self.Test_id       = processed_data['Test_df']['txkey']
        self.Record_list   = processed_data['Record_list']
        self.Cate_feat     = processed_data['Cate_feat']
        self.N_folds       = processed_data['N_folds']
        self.Target        = 'fraud_ind'
        self.Time          = datetime.today().strftime('%m-%d-%H-%M')

    def Cal_f1_score(self, True_y, Pre_y):
        return f1_score(True_y, Pre_y)
            
    def F1_score(self, y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        return 'f1', f1_score(y_true, y_hat), True
    
    def Modify_output(self, Ouptu_df):
        print('create modify output...')
        txkey_id = pd.read_csv('Attachment/mchno_fraud_ind.csv')
        Ouptu_df.loc[Ouptu_df.txkey.isin(txkey_id['txkey_id']), 'fraud_ind'] = 0
        return Ouptu_df

    def Submit(self, ids, preds):
        print('create submission csv file...')
        if not os.path.isdir('Submission'):
            os.makedirs('Submission')
        
        Filename= 'Submission/' + self.Time  + '.csv'
        submission = pd.DataFrame({'txkey': ids, 'fraud_ind': preds})
        submission.to_csv(Filename, index=False)
        #submission = self.Modify_output(submission)
        #submission.to_csv(Filename2, index=False)
        Processed_data['Record_list']['Filename'] = Filename

    def Split_kfolds(self):
        #print('K_folds ', self.N_folds)
        Train_X = self.Train_df.drop(['fraud_ind'], axis=1).values
        Train_Y = self.Train_df['fraud_ind'].values
        Folds  = KFold(n_splits=self.N_folds)
        Splited＿data = Folds.split(Train_X, Train_Y)     
        
        return Splited_data

    def Split_group_kfolds(self):
        Train_X = self.Train_df.drop(['fraud_ind'], axis=1)
        Train_Y = self.Train_df['fraud_ind']
        Folds  = GroupKFold(n_splits=self.N_folds)
        Splited_data = Folds.split(Train_X, Train_Y, groups=Train_X['Month']) 
        return Splited_data

    def display_importances(self, feature_importance_df):
        if not os.path.exists('Image'):
            os.mkdir('Image')

        cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

        plt.figure(figsize=(8, 10))
        ax = sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        fig = ax.get_figure()
        fig.savefig('Image/' + self.Time + '.png')

    def Save_record(self, Score):
        print('Initial Save_record....')
        if not os.path.exists('Record'):
            os.makedirs('Record')

        if not os.path.exists('Record/Record.csv'):
            Processed_data['Record_list']['Score'] = Score
            Result_df = pd.DataFrame(Processed_data['Record_list'])
            Result_df.to_csv('Record/Record.csv', index=False)

        else:
            Processed_data['Record_list']['Score'] = Score
            print(Processed_data['Record_list'])
            Result_df = pd.DataFrame(Processed_data['Record_list'])
            print('='*1000)
            History_result_df = pd.read_csv('Record/Record.csv')
            Result_df = pd.concat([History_result_df, Result_df], axis = 0)
            Result_df.to_csv('Record/Record.csv', index=False)

    def Under_sample(self, X, Y):
        Rand_under_sam = RandomUnderSampler(random_state=42)
        Feat = ['txkey']
        Sample_X, Sample_Y = Rand_under_sam.fit_resample(X[Feat], Y)
        Sample_Y = pd.DataFrame(Sample_Y, columns = ['fraud_ind'])
        Sample_X = pd.DataFrame(Sample_X, columns=['txkey'])
        Sample_X = pd.merge(Sample_X, X, on='txkey', how= 'inner')
        
        return Sample_X, Sample_Y
'''
    def Choose_threshold(self, pred):
        thresholds = [i*0.05 for i in range(1, 15)]
        f1_score_list = []
        for thres in thresholds:
            temp = np.where(pred > thres, 1, 0)
            self.Cal_f1_score()
'''

class LGBM(BaseModel):
    def Cv_train(self):
        print('LGBM training...')
        #Kfold
        #Splited_data = self.Split_kfolds()
        #GroupKfold
        Splited_data = self.Split_group_kfolds()
        
        Excluded_cols = ['fraud_ind', 'txkey', 'Month']
        All_Features = [Feat for Feat in self.Train_df.columns if Feat not in Excluded_cols]
       
        print('Total Feature : ', len(All_Features))
        print(All_Features)
        X = self.Train_df[All_Features +['Month']]
        Y = self.Train_df['fraud_ind']

        Oof = np.zeros(len(self.Train_df))
        Preds = np.zeros(len(self.Test_df))

        for fold_n, (train_idx, valid_idx) in enumerate(Splited_data):
            print(f'${fold_n+1} fold')
            X_trn, X_val= X[All_Features].iloc[train_idx], X[All_Features].iloc[valid_idx]
            Y_trn, Y_val = Y.iloc[train_idx], Y.iloc[valid_idx]
            print('Hold out ', X.iloc[valid_idx]['Month'].iloc[0], 'month')
            #X_trn, Y_trn = self.Under_sample(X_trn, Y_trn)
            
            #X_trn.drop('txkey', axis =1, inplace = True)
            #X_val.drop('txkey', axis =1, inplace = True)

            trn_data = lgb.Dataset(X_trn, label=Y_trn)
            val_data = lgb.Dataset(X_val, label=Y_val)
            
            clf = lgb.train(params= Lgb_params, train_set= trn_data, 
                                            valid_sets= [trn_data, val_data],
                                            num_boost_round = Lgb_params['num_boost_round'], 
                                            verbose_eval = Lgb_params['verbose_eval'], 
                                            early_stopping_rounds = Lgb_params['early_stopping_rounds'], 
                                            categorical_feature=self.Cate_feat, 
                                            feval=self.F1_score, 
                                            evals_result={})
            Y_pred_val = clf.predict(X_val)

            Oof[valid_idx] = Y_pred_val

            #All_Features.remove('txkey')
            Fold_importance_df = pd.DataFrame()
            Fold_importance_df['feature']    = All_Features
            Fold_importance_df['importance'] = np.log1p(clf.feature_importance(importance_type='gain', iteration=clf.best_iteration))
            Fold_importance_df['fold']       = fold_n + 1
            Fold_importance_df = pd.concat([Fold_importance_df, Fold_importance_df], axis=0)
            
            #print(f'Fold {fold_n+1} , F1: {self.Cal_f1_score(Y_val, Y_pred_val)}')

            #Score += self.Cal_f1_score(Y_val, Y_pred_val) / self.N_folds
            
            Preds += clf.predict(self.Test_df[All_Features]) / self.N_folds
            #All_Features.append('txkey')
        Oof = np.round(Oof)
        Preds = np.round(Preds)
        #Preds = np.where(Preds > 0.5, 1, 0)
        self.Submit(self.Test_df.txkey, Preds)
        Score = self.Cal_f1_score(Y, Oof)
        self.Save_record(Score)
        self.display_importances(Fold_importance_df)
        print(f' F1 = {self.Cal_f1_score(Y, Oof)}')
        return  Preds


class XGB(BaseModel):
    def Cv_train():
        pass

Lgb_model = LGBM(Processed_data)
