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
clf = SVC(gamma='scale', kernel = 'linear')
clf_rbf_auto = SVC(gamma = 'auto')
clf_rbf_scale = SVC(gamma = 'scale')
Y = np.array(clssfr)
Y = np.reshape(Y, (654,))
y = Y
X = np.array(predictors)
model_linear = clf.fit(X, Y) 
model_rbf_auto = clf_rbf_auto.fit(X,Y)
model_rbf_scale = clf_rbf_scale.fit(X,Y)

# Perform classification on samples in X.
y_prediction = clf.predict(X)


# Returns the mean accuracy on the given test data and labels.
linear_training_error = model_linear.score(X, Y)
print(linear_training_error)
#0.8516819571865444
rbf_score = model_rbf_auto.score(X,Y)
print(rbf_score)
#0.9938837920489296
rbf2_score = model_rbf_scale.score(X,Y)
print(rbf2_score)
#0.6605504587155964


# Weights assigned to the features
print('w = ',clf.coef_)

#Constant in the decision function
print('b = ',clf.intercept_)

###
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
