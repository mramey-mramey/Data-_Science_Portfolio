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
clf = SVC(gamma='auto', C= 0.0001, kernel = 'linear')
Y = np.array(clssfr)
Y = np.reshape(Y, (654,))
y = Y
X = np.array(predictors)
clf.fit(X, Y) 


# Perform classification on samples in X.
prediction = clf.predict(X)

# Returns the mean accuracy on the given test data and labels.
training_error = clf.fit(X, Y).score(X, Y)

params = clf.fit(X,Y).get_params()

# Weights assigned to the features
print('w = ',clf.coef_)

#Constant in the decision function
print('b = ',clf.intercept_)

###
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
