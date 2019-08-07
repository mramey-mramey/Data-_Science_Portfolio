import pandas as pd
import numpy as np
import glob
import sys
import time
import pickle
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sqlalchemy import types, create_engine
import cx_Oracle


# This script takes one YEAR argument
if len(sys.argv) != 3:
  print('Usage: ' + sys.argv[0] + 'Script name should be followed by MONTH and YEAR')
  sys.exit(1)
  
# Function loops thru each TRNSCTN TYP and creates a model to identify outliers  
def process_EE(whole_DF, each_TRNSCTN_TYP, outliers_fraction):
    df_features = whole_DF.loc[whole_DF['trnsctn_typ'] == each_TRNSCTN_TYP,'tot_trnsctn_amt']
    df_rest = whole_DF.loc[whole_DF['trnsctn_typ'] == each_TRNSCTN_TYP]
    X_train = df_features.values.reshape(-1,1)
    
    # load model
    filename = 'model_'+each_TRNSCTN_TYP+'.sav'
    envelope = pickle.load(open(filename, 'rb'))

    df_features = pd.DataFrame(df_features)
    df_features['deviation'] = envelope.decision_function(X_train)
    df_features['anomaly'] = envelope.predict(X_train)
    
    # combine
    df_final = pd.concat([df_features, df_rest], axis=1)
    return df_final




print('*****************Loading Data from Oracle ...*********************')
start = time.time()
YEAR = sys.argv[1]
MONTH = sys.argv[2]

conn = create_engine('oracle+cx_oracle://pkarunanidhi:Winter#2019@10.19.136.13:1526/subprd')

table = "select * from MSS_STG.SEL_TRNSCTN_HIST Where CTFF_DT = '30-APR-19'"

dfName = 'SEL_TRNSCTN_HIST_'+MONTH+YEAR+'_Anomaly.csv'

print('Starting Data Load')
SEL_TRNSCTN_HIST = pd.read_sql(table, con=conn)
conn.dispose()
endRead = time.time()
print(endRead - start)
print('To load data')

print('*****************Finish Loading ...*********************')


print('*****************Running Model ...*********************')

TRNSCTN_TYP_list = list(SEL_TRNSCTN_HIST['trnsctn_typ'].unique())
final_result = process_EE(SEL_TRNSCTN_HIST, TRNSCTN_TYP_list[0], 0.0001)
for i in range(1, len(TRNSCTN_TYP_list)-1):
    each_res = process_EE(SEL_TRNSCTN_HIST, TRNSCTN_TYP_list[i], 0.0001)
    final_result = pd.concat([each_res, final_result], ignore_index=True)

print('*****************Finish Running ...*********************')

anomaly = final_result.loc[final_result['anomaly'] == -1]
anomaly.to_csv(dfName,index=False)

print('*****************Save Anomalies to Local ...*********************')
print('*****************'+str(len(anomaly))+' Anomalies Found...*********************')

end = time.time()
print(end - start)
print('Program Finished')


















