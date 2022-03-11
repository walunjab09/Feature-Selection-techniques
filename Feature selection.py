# -*- coding: utf-8 -*-
"""
Feature selection methods:
    
    1. Filter methods : Filter each variable by statistical methods.
    
    2. Wrapper methods (Greedy methods): Finds out performance of each combination
       of features and selects the one with best performance.
       
    3. Embedded methods: Does feature selecting during model training
       eg. Random Forest, Lasso regression
       
"""

#%%  1. Univariate selection method : Filter method.

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('train.csv')

X = data.iloc[:,0:20]
y = data.iloc[:,-1]

best_features = SelectKBest(score_func = chi2, k = 10)
fit = best_features.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores = featureScores.sort_values(by = 'Score', ascending = False)

print(featureScores)


#%% 2. Feature importance method : Embedded method

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#%% 3. Correlation method

import seaborn as sns
corrmat = data.corr()

top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

