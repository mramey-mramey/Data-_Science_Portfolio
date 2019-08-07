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



conn_rfs_prd = create_engine('oracle+cx_oracle://pkarunanidhi:Winter#2019@10.19.136.13:1526/SUBPRD') 

query = "select * from MSS_STG.SEL_TRNSCTN_HIST Where CTFF_DT = '31-may-19'"

SEL_TRNSCTN_HIST = pd.read_sql(query, con = conn_rfs_prd)

df_head = SEL_TRNSCTN_HIST.head(n=10)

directory = (r'C:\Users\rk613ke\Desktop\2 - Ginnie\Python\SEL_TRNSCTN_HIST')
li = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"): 
        print(os.path.join(directory, filename))         
        fd = (os.path.join(directory, filename))
        df = pd.read_csv(fd, index_col=None, header=0)
        li.append(df)
        
        


SEL_TRNSCTN_HIST = pd.concat(li, ignore_index=True)
dfa = SEL_TRNSCTN_HIST.head()

df = SEL_TRNSCTN_HIST.head(n = 100)

y = df['TOT_TRNSCTN_AMT']
x = df['STG_TRNSCTN_ID']

#df = pd.read_excel(r'C:\Users\rk613ke\Desktop\2 - Ginnie\Python\CAR AD\CAR_TRNSCTN_HIST_Jan_2019_thru_Apr_2019.xlsx')




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



TRNSCTN_CD_list = list(SEL_TRNSCTN_HIST['TRNSCTN_CD'].unique())
final_result = process(SEL_TRNSCTN_HIST, TRNSCTN_CD_list[0], 0.0001)
for i in range(1, len(TRNSCTN_CD_list)-1):
    each_res = process(SEL_TRNSCTN_HIST, TRNSCTN_CD_list[i], 0.0001)
    final_result = pd.concat([each_res, final_result], ignore_index=True)

print('*****************Finish Running ...*********************')

dfName = r'C:\Users\rk613ke\Desktop\2 - Ginnie\Python\SEL_TRNSCTN_HIST__Anomaly_ALL.csv'
anomaly = final_result.loc[final_result['anomaly'] == -1]
anomaly.to_csv(dfName,index=False)

print('*****************Save Anomalies to Local ...*********************')
print('*****************'+str(len(anomaly))+' Anomalies Found...*********************')










########################### TEST CASE 2


df1 = SEL_TRNSCTN_HIST



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
df1 = SEL_TRNSCTN_HIST



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




'''




rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train2 = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()















'''

