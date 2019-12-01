import pandas as pd
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import gc
import numpy as np
from Tuning_parameter import Lgb_params

class Data():
    def __init__(self, Train_df, Test_df,N_folds):
        print('Data initailizing....')
        self.Train_df      = pd.read_csv('train.csv')
        self.Test_df       = pd.read_csv('test.csv')
        self.All_df        = pd.concat([self.Train_df, self.Test_df], axis=0, sort = True)
        self.Record_list   = {}
        self.Cate_feat     = []
        self.N_folds       = N_folds
        

    def Reduce_mem(self):
        start_mem_usg = self.All_df.memory_usage().sum() / (1024*1024)
        print("Memory usage of properties dataframe is :",start_mem_usg," MB")
        for col in self.All_df.keys():
            if self.All_df[col].dtype == int:
                Max = self.All_df[col].max()
                Min = self.All_df[col].min()
                if -128 < Min and Max < 127:
                    self.All_df[col] = self.All_df[col].astype(np.int8)
                elif -32768 < Min and Max < 32767:
                    self.All_df[col] = self.All_df[col].astype(np.int16)
                elif -2147483648 < Min and Max < 2147483647:
                    self.All_df[col] = self.All_df[col].astype(np.int32)
                else:
                    self.All_df[col] = self.All_df[col].astype(np.int64)

            elif self.All_df[col].dtype == float:
                self.All_df[col] = self.All_df[col].astype(np.float32)
            else:
                continue

        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = self.All_df.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    def Separate_dt(self):
        print('Initial Separate_dt.....')
        self.All_df = self.All_df[self.All_df.locdt < 120]
        
        for i in range(0,17):
            self.All_df.loc[self.All_df['locdt'] < 120 - 7*i, 'Month'] = 17 - i
        

        '''
        self.All_df['Month'] = 0
        self.All_df.loc[self.All_df['locdt'] < 121, 'Month'] = 4
        self.All_df.loc[(self.All_df['locdt']) < 91, 'Month']  = 3
        self.All_df.loc[(self.All_df['locdt']) < 61, 'Month']  = 2
        self.All_df.loc[(self.All_df['locdt']) < 31, 'Month'] = 1
        '''
        #self.All_df['Month'] = self.All_df['locdt'].apply(lambda x : x%6)

        self.Record_list['Separate_dt'] = 'Y'

    def Set_cate_feat(self, feat_list):
        print('Initial Set_cate_feat.....')
        #contp、etymd、stscd、flbmk、hcefg、flg_3dsmk
        self.Cate_feat = feat_list 

        self.Record_list['Set_cate_feat'] = [feat_list]
        
    def Label_encoding(self, feat_list):
        print('Initial Label_encoding.....')
        #insfg、ecfg、ovrlt、flbmk、flg_3dsmk
        le = LabelEncoder()
        for feat in feat_list:
            self.All_df[feat] = le.fit_transform(self.All_df[feat].astype(str))
            
        self.Record_list['Label_encoding'] = [feat_list]

    def Drop_feat(self, feat_list):
        print('Initial Drop_feat.....')
        self.All_df.drop(feat_list, axis=1, inplace = True)
        self.Record_list['Drop_feat'] = [feat_list]

    #FE
    def Process_txkey(self):
        print('Initial Process_txkey.....')
        #0.49 -> 0.55
        #Calculate count of transaction in the same account 
        self.All_df['Count_txkey_gb_bacno'] =  self.All_df.groupby(['bacno'])['txkey'].transform('count')
        #Calculate count of transaction in the same account and in the same card
        self.All_df['Count_txkey_gb_bacno'] =  self.All_df.groupby(['bacno', 'cano'])['txkey'].transform('count')
        #Calculate count of transaction in the same account in the same one hour
        self.All_df['Count_txkey_gb_bacno_locdt_Hour'] =  self.All_df.groupby(['bacno', 'locdt', 'Hour'])['txkey'].transform('count')
        

        #self.All_df['txkey_gb_stocn'] = self.All_df.groupby(['stocn'])['txkey'].transform('count')
        #self.All_df['txkey_gb_stocn_scity'] = self.All_df.groupby(['stocn'])['txkey'].transform('count')
        #self.All_df['Prop_stocn_scity'] = self.All_df['txkey_gb_stocn_scity'] / self.All_df['txkey_gb_stocn']  

        self.Record_list['Process_txkey'] = 'Y'

    def Process_cano(self):
        print('Initial Process_cano.....')
        #Calculate credit card in the same band account 
        self.All_df['Nuni_cano_gb_bacno'] = self.All_df.groupby(['bacno'])['cano'].transform('nunique')
        
        #Calculate count of transaction in the same account
        #self.All_df['Cnt_cano_gb_bacno'] = self.All_df.groupby(['bacno'])['cano'].transform('count')

        self.Record_list['Process_cano'] = 'Y'

    def Process_conam(self):
        self.All_df['Mean_conam_gb_etymd']   = self.All_df.groupby(['etymd'])['conam'].transform('mean')
        self.All_df['Medain_conam_gb_etymd'] = self.All_df.groupby(['etymd'])['conam'].transform('median')
        self.All_df['Std_conam_gb_etymd'] = self.All_df.groupby(['etymd'])['conam'].transform('std')
        self.All_df['Mean_conam_gb_csmcu']   = self.All_df.groupby(['csmcu'])['conam'].transform('mean')
        self.All_df['Medain_conam_gb_csmcu'] = self.All_df.groupby(['csmcu'])['conam'].transform('median')
        self.All_df['Std_conam_gb_csmcu'] = self.All_df.groupby(['csmcu'])['conam'].transform('std')
        self.All_df['Mean_conam_gb_cano'] = self.All_df.groupby(['cano'])['conam'].transform('mean')
        self.All_df['Medain_conam_gb_cano'] = self.All_df.groupby(['cano'])['conam'].transform('median')
        self.All_df['Std_conam_gb_cano'] = self.All_df.groupby(['cano'])['conam'].transform('std')
        self.All_df['Mean_conam_gb_mchno'] = self.All_df.groupby(['cano'])['mchno'].transform('mean')
        self.All_df['Medain_conam_gb_mchno'] = self.All_df.groupby(['cano'])['mchno'].transform('median')
        self.All_df['Std_conam_gb_mchno'] = self.All_df.groupby(['cano'])['mchno'].transform('std')

        self.All_df['Diff_mean_conam_gb_etymd']   = self.All_df['Mean_conam_gb_etymd'] - self.All_df['conam']
        self.All_df['Diff_median_conam_gb_etymd'] = self.All_df['Medain_conam_gb_etymd'] - self.All_df['conam']
        self.All_df['Diff_mean_conam_gb_etymd']   = self.All_df['Mean_conam_gb_csmcu'] - self.All_df['conam']
        self.All_df['Diff_median_conam_gb_etymd'] = self.All_df['Medain_conam_gb_csmcu'] - self.All_df['conam']
        self.All_df['Diff_mean_conam_gb_cano']   = self.All_df['Mean_conam_gb_cano'] - self.All_df['conam']
        self.All_df['Diff_median_conam_gb_cano'] = self.All_df['Medain_conam_gb_cano'] - self.All_df['conam']
        self.All_df['Diff_mean_conam_gb_mchno']   = self.All_df['Mean_conam_gb_mchno'] - self.All_df['conam']
        self.All_df['Diff_median_conam_gb_mchno'] = self.All_df['Medain_conam_gb_mchno'] - self.All_df['conam']

        self.All_df['Mean_conam_gb_bacno'] = self.All_df.groupby('bacno')['conam'].transform('mean')
        self.All_df['Medain_conam_gb_bacno'] = self.All_df.groupby('bacno')['conam'].transform('median')
        self.All_df['Diff_mean_conam_gb_bacno'] = self.All_df['Mean_conam_gb_bacno'] - self.All_df['conam']
        self.All_df['Diff_median_conam_gb_bacno'] = self.All_df['Medain_conam_gb_bacno'] - self.All_df['conam']

        #Calculate count of unique conam
        self.All_df['Cnt_conam_unique'] = self.All_df.groupby(['conam'])['acqic'].transform('count')
        self.All_df['Mean_conam'] = self.All_df['conam'] / self.All_df['iterm']
        self.Record_list['Process_conam'] = 'Y'
    
    def Process_csmcu(self):
        print('Initial Process_csmcu.....')
        Top5_array = self.All_df['csmcu'].value_counts().values[:10]
        self.All_df['csmcu'] = self.All_df['csmcu'].apply(lambda x : 0 if x not in Top5_array else x)
        self.Record_list['Process_csmcu'] = 'Y'

    def Process_flg_3dsmk(self):
        print('Initial flg_3dsmk.....')
        self.All_df.loc[self.All_df['flg_3dsmk'].isna(), 'flg_3dsmk_isna'] = 1
        self.All_df.loc[self.All_df['flg_3dsmk'].notna(), 'flg_3dsmk_isna'] = 0
        self.Record_list['Process_flg_3dsmk'] = 'Y'
        
    def Process_flbmk(self):
        print('Initial flbmk.....')
        self.All_df.loc[self.All_df['flbmk'].isna(), 'flbmk_isna'] = 1
        self.All_df.loc[self.All_df['flbmk'].notna(), 'flbmk_isna'] = 0
        
        self.Record_list['Process_flbmk'] = 'Y'

    def encode_CB(self, col1,col2):
        print('Initial encode_CB.....')
        new_col = col1+'_'+col2
        self.All_df[new_col] = self.All_df[col1].astype(str)+'_'+self.All_df[col2].astype(str)
        self.All_df[new_col] = self.All_df[col1].astype(str)+'_'+self.All_df[col2].astype(str) 

        le = LabelEncoder()
        self.All_df[new_col] = le.fit_transform(self.All_df[new_col].astype(str))
        
        self.Record_list['Process_encode_CB'] = 'Y'
        

    def Process_fre(self, cols):
        print('Initial Process_fre.....')
        for col in cols:
            vc = self.All_df[col].value_counts(dropna=True, normalize=True).to_dict()
            vc[-1] = -1
            new_col = col+'_freq'
            self.All_df[new_col] = self.All_df[col].map(vc)
            self.All_df[new_col] = self.All_df[col].astype('float32')

        self.Record_list['Process_flbmk'] = [cols]



    def Process_flg_3dsmk(self):
        print('Initial flg_3dsmk.....')
        self.All_df.loc[self.All_df['flg_3dsmk'].isna(), 'flg_3dsmk_isna'] = 1
        self.All_df.loc[self.All_df['flg_3dsmk'].notna(), 'flg_3dsmk_isna'] = 0
        self.Record_list['Process_flg_3dsmk'] = 'Y'

    def Process_iterm(self):
        print('Initial Process_iterm.....')
        self.All_df['0_iterm'] = 1
        self.All_df.loc[self.All_df['iterm'] == 0, '0_iterm'] = 0
        self.Record_list['Process_iterm'] = 'Y'
 

    def Process_acqic(self):
        print('Initial Process_acqic.....')
        Top5_array = self.All_df['acqic'].value_counts().values[:10]
        self.All_df['acqic'] = self.All_df['acqic'].apply(lambda x : 0 if x not in Top5_array else x)
        self.Record_list['Process_acqic'] = 'Y'

    def Process_loctm(self):
        print('Initial Process_loctm.....')
        def Str_turn_time(str1):
            str1 = str(int(str1))
            if len(str1) < 6:
                str1 = (6 - len(str1)) * '0' + str1

            return str1

        self.All_df['Hour'] = self.All_df['loctm'].apply(lambda x :Str_turn_time(x)[:2]).astype(int)
        self.All_df['Morning'] = 0
        self.All_df.loc[(self.All_df['Hour'].astype('int') > 7) & (self.All_df['Hour'].astype('int') < 22), 'Morning'] = 1
        
        self.Record_list['Process_loctm'] = 'Y'

        #self.All_df['Minute'] = self.All_df['loctm'].apply(lambda x :Str_turn_time(x)[2:4])
        #self.All_df['Second'] = self.All_df['loctm'].apply(lambda x :Str_turn_time(x)[4:6])
    
    def Process_mchno(self):
        print('Initial Process_mchno.....')
        self.All_df['Cnt_gb_machno'] = self.All_df.groupby(['mchno'])['iterm'].transform('count')

        #decrease mchno category feature nums
        Top5_array = self.All_df['mchno'].value_counts().values[:10]
        self.All_df['mchno_t'] = self.All_df['mchno'].apply(lambda x : 0 if x not in Top5_array else x)

        self.Record_list['Cnt_gb_machno'] = 'Y'

    def Process_locdt(self):
        print('Initial Process_locdt.....')
        self.All_df['Week'] = self.All_df['locdt'].apply(lambda x : x%7)
        self.Record_list['Process_locdt'] = 'Y'
        
    def One_hot(self, list1):
        print('Initial One_hot.....')
        #contp、flbmk、flg_3dsmk、hcefg、insfg、stscd
        for col in list1:
            one_hot = pd.get_dummies(self.All_df[col], prefix = col)
            gc.collect()
            self.All_df.drop(col, inplace = True, axis =1)
            self.All_df = pd.concat([self.All_df, one_hot], axis =1)
        self.Record_list['Process_loctm'] = [list1]

    def Process_mcc(self):
        print('Initial Process_mcc.....')
        Top5_array = self.All_df['mcc'].value_counts().values[:10]
        self.All_df['mcc'] = self.All_df['mcc'].apply(lambda x : 0 if x not in Top5_array else x)


    def Return_record(self):
        print('Initial Return_record.....')
        self.Train_df = self.All_df[:1521787]
        #self.Train_df = self.Train_df[self.Train_df['Date_group'] == 3]
        self.Test_df  = self.All_df[1521787:]
        
        #Add Parameters
        self.Record_list['Lgb_parama'] = [Lgb_params]

        Final_data = {
            'Train_df'    : self.Train_df,
            'Test_df'     : self.Test_df,
            'Record_list' : self.Record_list,
            'Cate_feat'   : self.Cate_feat,
            'N_folds'     : self.N_folds,
            #'Features'    : [Feat for Feat in self.Train_df if Feat not in [self.Cate_feat] and Feat not in ['fraud_ind']]
        } 
        
        return Final_data

Train_df = pd.read_csv('train.csv')
Test_df  = pd.read_csv('test.csv')
data = Data(Train_df, Test_df, 5)
data.Reduce_mem()
data.Separate_dt()
data.Label_encoding(['insfg', 'ecfg', 'ovrlt', 'flbmk', 'flg_3dsmk'])
data.Process_csmcu()
#contp、flbmk、flg_3dsmk、hcefg、insfg、stscd
#data.One_hot(['flbmk', 'flg_3dsmk', 'hcefg', 'insfg, 'stscd'])
cate_feats = ['etymd', 'acqic', 'mcc', 'contp', 'flbmk', 'flg_3dsmk', 'hcefg', 'insfg', 'stscd', 'mchno']
data.Set_cate_feat(cate_feats)
data.Process_cano()
data.encode_CB('stocn', 'scity')
data.Process_iterm()
data.Process_acqic()
data.Process_loctm()
data.Process_txkey()
#data.Process_conam()
data.Process_mchno()
#data.Process_locdt()
data.Process_flg_3dsmk() 
data.Process_flbmk()
data.Process_fre(cate_feats + ['stocn_scity'])
#data.Drop_feat(['locdt'])
data.Drop_feat(['locdt', 'bacno', 'cano', 'stocn', 'scity'])
Processed_data = data.Return_record()



