import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools

#loading the cancer data from csv file
cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head())

#the distribution of the classes based on the Clump thickness and uniformity of cell size
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

#data preprocessing and selection
print(cell_df.dtypes)

#the BareNuc column includes some values that are not numerical, dropping that row
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

#selecting the feature and target dataset
X=cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']].values
print(X[0:5])
y=cell_df[['Class']].values
print(y[0:5])

#spliting the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#fitting the data into svm model
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

#predicting the ouput of test dataset
yhat = clf.predict(X_test)
print(yhat [0:5])

'''EVALUATION'''

#jacard index
print('The jacard index ',jaccard_similarity_score(y_test, yhat))

#confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

#the classification report
print ('The classifiaction report \n',classification_report(y_test, yhat))

#f1 score
print('the f2 score ',f1_score(y_test, yhat, average='weighted'))
