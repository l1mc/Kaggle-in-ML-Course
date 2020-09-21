# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:16:45 2020

@author: Mingcong Li
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


mpl.rcParams['agg.path.chunksize'] = 10000
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)


# define a function to count how many unique valus in each column and to show how many times each unique value appears.
def describe_data(variablename):
    unique_value = list(variablename.unique())   # unique value in each column
    print('There are %i unique values in this column.' %len(unique_value))
    print('The frequency of each value is shown below:')
    print(pd.DataFrame(variablename.value_counts()).sort_index()) # how many time each unique value appears
    return  pd.DataFrame(variablename.value_counts()).sort_index()




# import data
train_df = pd.read_csv(r'D:\machine-learning-for-business-classification\dataframe_train.csv')
test_df = pd.read_csv(r'D:\machine-learning-for-business-classification\dataframe_test.csv')
classification_df = pd.read_csv(r'D:\machine-learning-for-business-classification\Classification.csv',index_col=0)
regression_df = pd.read_csv(r'D:\machine-learning-for-business-regression\Regression.csv',index_col=0)
# combine these datasets to run certain operations on both datasets together
combine = [train_df, test_df]

# describe data
# training set
print(train_df.columns.values)  # see column names
train_df.head(1).T  # see sample data
train_df.tail(1).T
train_df.info()  # see core information
train_df.describe()
train_df['date']  # see one column
train_df.query('courier_id==10007871')
describe_data(test_df['source_type'])  # describe each column
# testing set
test_df.info()
print(test_df.columns.values)
test_df.describe()



# Coding of discrete variables
# There is no difference in the value of discrete features, which are action_type, weather_grade, and source_type. Use one-hot encoding.
# training set
a_t = pd.get_dummies(train_df['action_type']).iloc[:,0] #  only take DELIVERY
a_t = a_t.rename('delivery_AT') # series. rename to make difference with source_type
w_g = pd.get_dummies(train_df['weather_grade'])
s_t = pd.get_dummies(train_df['source_type'])
s_t.columns=['assign_ST','delivery_ST','pickup_ST']
train_df_concat = pd.concat([train_df,a_t,w_g,s_t],join='outer', axis=1)
train_df_concat.info()
train_df_concat.head(1).T
# testing set
w_g = pd.get_dummies(test_df['weather_grade'])
s_t = pd.get_dummies(test_df['source_type']).iloc[:,1:]
s_t.columns=['assign_ST','delivery_ST','pickup_ST']
test_df_concat = pd.concat([test_df,w_g,s_t],join='outer', axis=1)
test_df_concat.info()
test_df_concat.head(1).T


# cleaning data
# codes to show the character of a variable
describe_data(train_df_concat['grid_distance']).iloc[-90:,:]

# Remove extreme outliers
train_df_concat.drop(train_df_concat[train_df_concat['grid_distance'] > 10000].index, inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(train_df_concat['grid_distance'], bins=300)
plt.show()


# creat Xtrain, Ytrain
# training set
Xtrain = train_df_concat.drop(train_df_concat[['action_type','weather_grade','source_type','expected_use_time','delivery_AT','aoi_id','shop_id']], axis=1)
Xtrain.head(1).T
Xtrain.columns.values
Ytrain = train_df_concat.iloc[:,-8]
# validation set
Xtrain, Xvalid, Ytrain, Yvalid =train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=42)
# testing set
Xtest = test_df_concat.drop(test_df_concat[['weather_grade','source_type','aoi_id','shop_id']], axis=1)
Xtest.head(1).T



# method1: Decision Tree -- score on Kaggle = 0.75
max_features=['auto', 'sqrt', 'log2',None]
clf=tree.DecisionTreeClassifier(criterion="gini",random_state=30,splitter="random",max_depth=20,min_samples_leaf=250,min_samples_split=1000,max_features=max_features[3])
clf=clf.fit(Xtrain,Ytrain)

# test the tree on validation set
# the mean accuracy on the given test data and labels
score=clf.score(Xvalid,Yvalid)  # score
print(score)  # 0.7803089767375482
# f1 score
y_pred = clf.predict(Xvalid)  # make prediction
f1 = f1_score(Yvalid, y_pred, average='binary')  # f1 score
print(f1)  # 0.7985341992900028

# information about the tree
clf.get_depth()  # the depth of the tree
[*zip(Xtrain.columns.values,clf.feature_importances_)]  # the importance of each feature

# cope with testing set
Xtest.to_csv('Xtest.csv')
Xtest=pd.read_csv('Xtest.csv',index_col=0)
Xtest.columns.values

# predict
y_pred = clf.predict(Xtest)
classification_df['action_type_DELIVERY']=y_pred
classification_df.to_csv(r'D:\machine-learning-for-business-classification\Classification-Mingcong2.csv')



# Method 2: Random Forest --- score on Kaggle = 0.79446
# fit
rfc = RandomForestClassifier(n_estimators=75, criterion='gini', max_depth=50, min_samples_split=25, n_jobs=-1, random_state=0)
rfc=rfc.fit(Xtrain,Ytrain)

# judge
scorer=rfc.score(Xvalid,Yvalid)
print(score)
y_pred = rfc.predict(Xvalid)  # make prediction
f1 = f1_score(Yvalid, y_pred, average='binary')
print(f1)

# predict
y_pred = rfc.predict(Xtest)
classification_df['action_type_DELIVERY']=y_pred
classification_df.to_csv(r'D:\machine-learning-for-business-classification\Classification-Mingcong-RF.csv')



# method 3: XGBoost ---  score on Kaggle = 0.89806, I chose this model
# load data
dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dvalid = xgb.DMatrix(Xvalid, label=Yvalid)
dtest = xgb.DMatrix(Xtest)
# Setting Parameters
param = {'max_depth': 15, 'eta': 0.2, 'objective': 'binary:logistic'}
param['nthread'] = 12
param['eval_metric'] = 'auc'
evallist = [(dvalid, 'eval'), (dtrain, 'train')]
# Training
num_round = 100
bst = xgb.train(param, dtrain, num_round, evallist)
# judge
ypred = bst.predict(dvalid)
ypred=ypred>0.5
ypred=ypred+0
f1 = f1_score(Yvalid, ypred, average='binary')
print(f1)
# predict
y_pred = bst.predict(dtest)
y_pred=y_pred>0.5
y_pred=y_pred+0
classification_df['action_type_DELIVERY']=y_pred
classification_df.to_csv(r'D:\machine-learning-for-business-classification\Classification-Mingcong-XGB.csv')





# Q2: regression
Ytrain = train_df_concat.iloc[:,-11]
Xtrain, Xvalid, Ytrain, Yvalid =train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=42)

#
# load data
dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dvalid = xgb.DMatrix(Xvalid, label=Yvalid)
dtest = xgb.DMatrix(Xtest)

# Setting Parameters
param = {'max_depth': 15, 'eta': 0.05, 'objective': 'reg:linear'}
param['nthread'] = 12
param['eval_metric'] = 'mae'
param['booster'] = 'gbtree'
evallist = [(dvalid, 'eval'), (dtrain, 'train')]
# Training
num_round = 41
bst = xgb.train(param, dtrain, num_round, evallist)

# predict
y_pred = bst.predict(dtest)
regression_df['expected_use_time']=y_pred
regression_df.to_csv(r'D:\machine-learning-for-business-regression\Regression-Mingcong-XGB.csv')


















# learning curve of hyperparameter
superpa = []
for i in range(20,210,10):
    rfc = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=20, min_samples_split=25, n_jobs=-1, random_state=0)
    rfc=rfc.fit(Xtrain,Ytrain)
    y_pred = rfc.predict(Xvalid)  # make prediction
    f1 = f1_score(Yvalid, y_pred, average='binary')
    superpa.append(f1)
print(max(superpa),superpa.index(max(superpa)))
plt.figure(figsize=[20,5])
plt.plot(range(20,210,10),superpa)
plt.show()


# This action takes too long time
# grid search
# set parameters
parameters = {'splitter':('best','random'),'criterion':("gini","entropy"),"max_depth":[*range(3,14,5)],'min_samples_leaf':[*range(1,10,4)],'min_impurity_decrease':[*np.linspace(0,0.5,3)]}
# search
clf = tree.DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, scoring='f1', n_jobs=-1,cv=3)
GS.fit(Xtrain,Ytrain)
GS.best_params_
GS.best_score_



