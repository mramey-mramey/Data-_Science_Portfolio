import numpy as np
import matplotlib.pyplot as plt
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


import pandas as pd


file_loc = (r'C:\Users\Mitchell.Ramey\Documents\credit_card_data-headers.txt')
file = open(file_loc)
file = file.read()
df = pd.read_csv(file_loc, delim_whitespace=True)

#Seperating the Classifier/Predictors
clssfr = df.iloc[:,10:]
clssfr2 = df.loc[:,'R1']
predictors = df.loc[:,'A1':'A15']

X = np.array(predictors)
y = np.array(clssfr)
y = np.reshape(y,(654,))



# splitting the data into training and test sets (80:20)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

print('X_train shape = '+ str(X_train.shape))
print('X_test shape = ' + str(X_test.shape))

print('y_train shape = ' + str(y_train.shape))
print('y_test shape = '+ str(y_test.shape))


#Try running from k=1 through 25 and record testing accuracy
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))

#Testing accuracy for each value of K        
print(scores)


#plot the relationship between K and the testing accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


#Making prediction on some unseen data 
#predict for the below two random observations
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X,y)
classes = {0:'denied',1:'approved'}

x_new = X_train[1:50]
y_predict = knn.predict(x_new)

xnl = len(x_new)

for i in range(0,xnl):
    print(classes[y_predict[i]])
