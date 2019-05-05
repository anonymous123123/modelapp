import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection,metrics
import pickle


def transformDataFrame(df):
    keep_features = ['status_last_archived_0_24m', 'num_arch_ok_0_12m', 'status_3rd_last_archived_0_24m', 'account_worst_status_0_3m', 'num_unpaid_bills', 'status_max_archived_0_24_months', 'num_arch_ok_12_24m', 'age', 'num_active_div_by_paid_inv_0_12m', 'avg_payment_span_0_12m', 'max_paid_inv_0_24m', 'status_2nd_last_archived_0_24m', 'account_status', 'merchant_group', 'sum_paid_inv_0_12m', 'max_paid_inv_0_12m', 'time_hours']
    df=df[keep_features]
    df = transformMerchantGroup(df)
    df = transformCategorical(df)
    df = transformNumerical(df)
    df['age^2']=df['age']**2

    column_names = df.columns.tolist()
    column_names.sort()
    return df[column_names]


def transformMerchantGroup(df):
    high_risk = ['Food & Beverage','Intangible products']
    df['high_risk_merchant_group']=0
    df.loc[df['merchant_group'].isin(high_risk),'high_risk_merchant_group']=1
    df = df.drop(['merchant_group'],axis=1,inplace=False)
    return df

def transformCategorical(df):
    dummy_values = {'status_max_archived_0_24_months':[1.0,2.0,3.0,5.0], 'account_status':[2.0,3.0,4.0], 'status_last_archived_0_24m':[1.0,2.0,3.0,5.0], 'status_2nd_last_archived_0_24m':[1.0,2.0,3.0,5.0], 'status_3rd_last_archived_0_24m':[1.0,2.0,3.0,5.0], 'account_worst_status_0_3m':[2.0,3.0,4.0]}

    for categorical in dummy_values:
        dummy_name =categorical+"_nan"
        df[dummy_name]=0
        df.loc[df[categorical].isna(),dummy_name]=1
        for value in dummy_values[categorical]:
            dummy_name =categorical+"_"+str(value)
            df[dummy_name]=0
            df.loc[df[categorical]==value,dummy_name]=1
        df = df.drop([categorical],axis=1,inplace=False)
    return df

def transformNumerical(df):
    medians ={'num_active_div_by_paid_inv_0_12m':0.0, 'age':34.0, 'avg_payment_span_0_12m':14.909091, 'max_paid_inv_0_24m':7580.0, 'sum_paid_inv_0_12m':15995.0, 'num_arch_ok_12_24m':2.0, 'time_hours':15.792778, 'num_unpaid_bills':0.0, 'max_paid_inv_0_12m':6052.0, 'num_arch_ok_0_12m':2.0}

    for feature in medians:
        df.loc[df[feature].isna(),feature]=medians[feature]

    return df

def over_sampling(X,y):
    df = pd.concat([X,y],axis=1)

    count_class_0, count_class_1 = df.default.value_counts()

    df_class_0 = df[df['default'] == 0]
    df_class_1 = df[df['default'] == 1]

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    y = df_test_over['default']
    X = df_test_over.drop(['default'],axis=1,inplace=False)

    return X,y


df = pd.read_csv('/Users/eric.ohman/MLCaseStudy/dataset.csv',sep=";")

#assignment_test_df = df.loc[df['default'].isnull()]
train_df = df.loc[df['default'].notnull()]

gb_classifier = GradientBoostingClassifier()

train_y= train_df['default']
train_X = transformDataFrame(train_df)

X_sm, y_sm = over_sampling(train_X, train_y)
gb_classifier.fit(X_sm,y_sm)


filename = 'gb_classifier.sav'
pickle.dump(gb_classifier, open(filename, 'wb'))

'''
y_preds = pd.DataFrame()

ypred = gb_classifier.predict_proba(assignement_input)[::,1]
y_preds = y_preds.append(pd.DataFrame(data=ypred))

assignment_answer['pd'] = y_preds.values


assignment_answer.to_csv (r'/Users/eric.ohman/MLCaseStudy/predictions_local.csv', index = None, header=True,sep=";")
'''