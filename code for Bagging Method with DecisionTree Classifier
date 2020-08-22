Importing libraries
from sklearn.tree importimport seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
import os
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score ,mean_squared_error
from math import sqrt
from sklearn.ensemble import BaggingClassifier 


Loading the data 
data = pd.read_csv('../input/prediction/fer (1).csv')  
      
    # Printing the dataswet shape 
print ("Dataset Length: ", len(data)) 
print ("Dataset Shape: ", data.shape) 
      
    # Printing the dataset obseravtions 
print ("Dataset: ",data.head(10)) 
 
    # Separating the target variable 
X = data.values[:, 1:12] 
Y = data.values[:, 13] 

# load the data 
kfold = model_selection.KFold(n_splits = 5, 
                       random_state = None) 

# initialize the base classifier 
base_cls = DecisionTreeClassifier() 

# no. of base classifier 
num_trees = 50


# Calculating Accuracy  with bagging classifier 

model= BaggingClassifier(base_estimator = base_cls, 
            random_state = 10,n_estimators = num_trees) 
results = model_selection.cross_val_score(model, X, Y, cv = kfold) 
results=results.mean()
print("Accuracy with Bagging Classifier:",results*100) 



#Calculating cohen score with Bagging classifier and with decision tree as a base estimator 

from sklearn.metrics import cohen_kappa_score

model.fit(X_train,y_train)
model.score(X_test,y_test)
Y_pred = model.predict(X_test)
cohen_score = cohen_kappa_score(y_test, Y_pred)
print("Cohen's Kappa index ||Decision Tree with bagging:" ,cohen_score)

#RSME( Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
print("RMSE using Bagging classifier:",mean_squared_error(y_test, y_pred))



