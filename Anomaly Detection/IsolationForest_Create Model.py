import cx_Oracle
from sqlalchemy import types, create_engine
import pandas as pd
import time
import xlsxwriter 
import numpy as np 
import os 
import string
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import glob
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib
import seaborn as sns
import matplotlib.dates as md
from matplotlib import pyplot as plt



conn_rfs_prd = create_engine('oracle+cx_oracle://') 

query = ""

TRNSCTN_HIST = pd.read_sql(query, con = )

df_head = .head(n=10)

directory = (r'')
li = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"): 
        print(os.path.join(directory, filename))         
        fd = (os.path.join(directory, filename))
        df = pd.read_csv(fd, index_col=None, header=0)
        li.append(df)
        
        


TRNSCTN_HIST = pd.concat(li, ignore_index=True)
dfa = TRNSCTN_HIST.head()

df = TRNSCTN_HIST.head(n = 100)

y = df['TOT_TRNSCTN_AMT']
x = df['STG_TRNSCTN_ID']

#df = pd.read_excel(r'')




def process(whole_DF, each_TRNSCTN_CD, outliers_fraction):
    df_features = whole_DF.loc[whole_DF['TRNSCTN_CD'] == each_TRNSCTN_CD,'TOT_TRNSCTN_AMT']
    df_rest = whole_DF.loc[whole_DF['TRNSCTN_CD'] == each_TRNSCTN_CD]
    X_train = df_features.values.reshape(-1,1)
    xtl = len(X_train)
    # load model
    clf = IsolationForest(behaviour='new', contamination= outliers_fraction, n_estimators = 50, max_samples = xtl)
    clf.fit(X_train)
    
    envelope =  EllipticEnvelope(contamination=outliers_fraction, support_fraction=1)
    envelope.fit(X_train)
    
    filename_IF = 'model_IF_'+each_TRNSCTN_CD+'.sav'
    pickle.dump(clf, open(filename_IF, 'wb'))
    
    filename_EE = 'model_EE_'+each_TRNSCTN_CD+'.sav'
    pickle.dump(envelope, open(filename_EE, 'wb'))

    return df_rest



TRNSCTN_CD_list = list(TRNSCTN_HIST['TRNSCTN_CD'].unique())
final_result = process(TRNSCTN_HIST, TRNSCTN_CD_list[0], 0.0001)
for i in range(1, len(TRNSCTN_CD_list)-1):
    each_res = process(TRNSCTN_HIST, TRNSCTN_CD_list[i], 0.0001)
    final_result = pd.concat([each_res, final_result], ignore_index=True)

print('*****************Finish Running ...*********************')

dfName = r'C:\Users\rk613ke\Desktop\2 - Ginnie\Python\TRNSCTN_HIST__Anomaly_ALL.csv'
anomaly = final_result.loc[final_result['anomaly'] == -1]
anomaly.to_csv(dfName,index=False)

print('*****************Save Anomalies to Local ...*********************')
print('*****************'+str(len(anomaly))+' Anomalies Found...*********************')










########################### TEST CASE 2


df1 = TRNSCTN_HIST



df2 = df1.loc[:,'tot_trnsctn_amt']

df2 = df2.replace({ np.nan:0})

X_train = df2.values.reshape(-1,1)

xtl = len(X_train)

rng = np.random.RandomState(42)

clf = IsolationForest(behaviour='new', contamination= 0.0001, n_estimators = 100, max_samples = xtl)
clf.fit(X_train)



df2 = pd.DataFrame(df2)

df2['anomaly'] = clf.predict(X_train)


df_final = pd.concat([df1, df2], axis=1)

anomaly = df_final.loc[df_final['anomaly'] == -1]











#Local Outlier Factor
df1 = TRNSCTN_HIST



df2 = df1.loc[:,'tot_trnsctn_amt']

df2 = df2.replace({ np.nan:0})

X = df2.values.reshape(-1,1)

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.0001)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).


df2 = pd.DataFrame(df2)

df2['anomaly'] = clf.fit_predict(X)

df_final = pd.concat([df1, df2], axis=1)

anomaly2 = df_final.loc[df_final['anomaly'] == -1]






