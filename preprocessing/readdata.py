import pandas as pd
import numpy as np


class read_csv:
    def read_data():
        nvals = 1000   # Change to None for Reading all data
        #Read Bureau Balance data and transform to merge with App_train 
        bureau_bal = pd.read_csv('../data/bureau_balance.csv',nrows=nvals, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        bureau_bal = pd.concat([bureau_bal, pd.get_dummies(bureau_bal.STATUS, prefix='bureau_bal_status')], axis=1).drop('STATUS', axis=1)
        bureau_counts = bureau_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
        bureau_bal['bureau_count'] = bureau_bal['SK_ID_BUREAU'].map(bureau_counts['MONTHS_BALANCE'])
        avg_bureau_bal = bureau_bal.groupby('SK_ID_BUREAU').mean()
        avg_bureau_bal.columns = ['avg_' + val for val in avg_bureau_bal.columns]

        #Read Bureau Data Transform and Merge with Bureau Balance Data
        bureau = pd.read_csv('../data/bureau.csv',nrows=nvals, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        credit_act = pd.get_dummies(bureau.CREDIT_ACTIVE, prefix='ca_')
        credit_curr = pd.get_dummies(bureau.CREDIT_CURRENCY, prefix='cu_')
        credit_type = pd.get_dummies(bureau.CREDIT_TYPE, prefix='ty_')
        bureau_ct = pd.concat([bureau, credit_act, credit_curr, credit_type], axis=1)
        bureau_merged = bureau_ct.merge(right=avg_bureau_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bureau_bal'))
        bureau_per_count = bureau_merged[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
        bureau_merged['SK_ID_BUREAU'] = bureau_merged['SK_ID_CURR'].map(bureau_per_count['SK_ID_BUREAU'])
        bureau_avg = bureau_merged.groupby('SK_ID_CURR').mean()

        # Read Previous Application Data and Transform
        prev_app = pd.read_csv('../data/previous_application.csv',nrows=nvals, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        categorical_feats = [ cat for cat in prev_app.columns if prev_app[cat].dtype == 'object']
        categorical_feats = categorical_feats[2:]
        #Factarozise Previous Application data there are so many Object Columns 
        for cat in categorical_feats:
            prev_app[cat],indexer = pd.factorize(prev_app[cat])
        prev_app_count = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
        prev_app['SK_ID_PREV'] = prev_app['SK_ID_CURR'].map(prev_app_count['SK_ID_PREV'])
        prev_apps_avg = prev_app.groupby('SK_ID_CURR').mean()
        prev_apps_avg.columns = ['prev_' + col for col in prev_apps_avg.columns]
        
        #Read Cash Balance and Transform to merge with App Train Data
        pos_cash_bal = pd.read_csv('../data/POS_CASH_balance.csv',nrows=nvals, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        pos_bal = pd.concat([pos_cash_bal, pd.get_dummies(pos_cash_bal['NAME_CONTRACT_STATUS'])], axis=1)
        pos_bal_count = pos_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
        pos_bal['SK_ID_PREV'] = pos_bal['SK_ID_CURR'].map(pos_bal_count['SK_ID_PREV'])
        avg_pos_bal = pos_bal.groupby('SK_ID_CURR').mean()
        
        
        #Read Credit Balance and Transform to merger with App Train Data
        credit_bal = pd.read_csv('../data/credit_card_balance.csv',nrows=nvals, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        credit_bal = pd.concat([credit_bal, pd.get_dummies(credit_bal['NAME_CONTRACT_STATUS'], prefix='credit_status_')], axis=1)
    
        credit_bal_count = credit_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
        credit_bal['SK_ID_PREV'] = credit_bal['SK_ID_CURR'].map(credit_bal_count['SK_ID_PREV'])
        avg_credit_bal = credit_bal.groupby('SK_ID_CURR').mean()
        avg_credit_bal.columns = ['credit_bal_' + f_ for f_ in avg_credit_bal.columns]
        
        # Read Intsallment Payments data and Transform for merging with App Train Data
        inst_pay = pd.read_csv('../data/installments_payments.csv',nrows=nvals, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        # As Install Payments as many Object columns get the features and factorize
        categorical_feats = [ cat for cat in inst_pay.columns if inst_pay[cat].dtype == 'object']
        categorical_feats = categorical_feats[2:]
        for cat in categorical_feats:
            inst_pay[cat],indexer = pd.factorize(inst_pay[cat])


        inst_pay_counts = inst_pay[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
        inst_pay['SK_ID_PREV'] = inst_pay['SK_ID_CURR'].map(inst_pay_counts['SK_ID_PREV'])
        avg_inst_pay = inst_pay.groupby('SK_ID_CURR').mean()
        avg_inst_pay.columns = ['inst_' + f_ for f_ in avg_inst_pay.columns]
        

        # Read Application Training Data and Factorize as many Onject Columns

        app_train = pd.read_csv('../data/application_train.csv',nrows=nvals, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')         
        y = app_train['TARGET']
        del app_train['TARGET']
        categorical_feats = [ cat for cat in app_train.columns if app_train[cat].dtype == 'object']
        categorical_feats = categorical_feats[1:]
        for cat in categorical_feats:
            app_train[cat],indexer = pd.factorize(app_train[cat])
        # Merger with all the Avg Dara by SKID
        app_train_final = app_train.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')
        app_train_final = app_train_final.merge(right=avg_credit_bal.reset_index(), how='left', on='SK_ID_CURR')
        app_train_final = app_train_final.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')
        app_train_final = app_train_final.merge(right=avg_pos_bal.reset_index(), how='left', on='SK_ID_CURR')
        app_train_final = app_train_final.merge(right=avg_inst_pay.reset_index(), how='left', on='SK_ID_CURR')
        app_train_final = app_train_final.fillna(0)
        return (app_train_final , y)