from sklearn.svm import SVC
import pandas as pd
import numpy as np

#Reading in the data
file_loc = (r'C:\Users\Mitchell.Ramey\Documents\credit_card_data-headers.txt')
file = open(file_loc)
file = file.read()
df = pd.read_csv(file_loc, delim_whitespace=True)

#Seperating the Classifier/Predictors
clssfr = df.iloc[:,10:]
clssfr2 = df.loc[:,'R1']
predictors = df.loc[:,'A1':'A15']


#Fitting the model
clf = SVC(gamma='auto', C= 0.0001)
Y = np.array(clssfr)
Y = np.reshape(Y, (654,))
X = np.array(predictors)
clf.fit(X, Y) 

# Perform classification on samples in X.
prediction = clf.predict(X)

# Returns the mean accuracy on the given test data and labels.
training_error = clf.fit(X, Y).score(X, Y)