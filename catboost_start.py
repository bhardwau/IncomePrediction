# Importing the libraries
import numpy as np
import math
import pandas as pd 
import lightgbm

# Importing the dataset
dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv')
submit = pd.read_csv('tcd-ml-1920-group-income-test.csv')

dataset = dataset.fillna(0)
submit = submit.fillna(0)

dataset['Total Yearly Income [EUR]'] = np.log(dataset['Total Yearly Income [EUR]'])

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].values

# dataset.corr(method="spearman")

# X = X.drop(['Housing Situation'], axis = 'columns')
# submit = submit.drop(['Housing Situation'], axis = 'columns')

X['Work Experience in Current Job [years]'].replace(['#NUM!'], [0], inplace = True)
submit['Work Experience in Current Job [years]'].replace(['#NUM!'], [0], inplace = True)

X.Gender.replace(['0', 'unknown', 'f'], ['unknown' , 'unknown', 'female'], inplace=True)
submit.Gender.replace(['0', 'unknown'], ['unknown' , 'unknown'], inplace=True)


X['University Degree'].replace(['0', 0], ['No', 'No'], inplace = True)
submit['University Degree'].replace(['0', 0], ['No', 'No'], inplace = True)

X = X.drop(['Instance', 'Yearly Income in addition to Salary (e.g. Rental Income)'] , axis='columns')
submit = submit.drop(['Instance','Yearly Income in addition to Salary (e.g. Rental Income)'] , axis='columns')

X["Work Experience in Current Job [years]"] = (X["Work Experience in Current Job [years]"].astype(float))
X["Work Experience in Current Job [years]"] = np.ceil((X["Work Experience in Current Job [years]"]))
X["Work Experience in Current Job [years]"] = (X["Work Experience in Current Job [years]"].astype(int))

submit["Work Experience in Current Job [years]"] = (submit["Work Experience in Current Job [years]"].astype(float))
submit["Work Experience in Current Job [years]"] = np.ceil((submit["Work Experience in Current Job [years]"]))
submit["Work Experience in Current Job [years]"] = (submit["Work Experience in Current Job [years]"].astype(int))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from catboost import CatBoostRegressor
model=CatBoostRegressor(task_type = 'GPU', iterations = 1000, learning_rate = 0.05)
#categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(X_train,y_train ,cat_features=([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]), eval_set=(X_test, y_test))
model.score(X_test, y_test)

ans = model.predict(submit)

ans = np.exp(ans)
