import numpy as np
import pandas as pd
import pickle

data = pd.read_csv('train.csv')
# print(data.info())

# for column in data.columns:
#     print(data[column].value_counts())
#     print("*"*20)

# print(data.isna().sum())

data.drop(columns=['lot_size', 'lot_size_units'], inplace=True)

data['price_per_sqft'] = data['price']*100000/data['size']
print(data['price_per_sqft'])
data.to_csv('final_dataset.csv')
X = data.drop(columns=['price', 'size_units'])
Y = data['price']
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['beds']), remainder='passthrough')
scaler = StandardScaler()

lr = LinearRegression()
X_scaled = scaler.fit_transform(X)

lr.fit(X_scaled, Y)

pipe = make_pipeline(column_trans, scaler, lr)
pipe.fit(X_train, y_train)
y_pred_lr = pipe.predict(X_test)
print("No Regularization: ", r2_score(y_test, y_pred_lr))

# **********************************************

lasso = Lasso()
pipe = make_pipeline(column_trans, scaler, lasso)
pipe.fit(X_train, y_train)
y_pred_lasso = pipe.predict(X_test)
print("Lasso: ", r2_score(y_test, y_pred_lasso))

# ************************************************

ridge = Ridge()
pipe = make_pipeline(column_trans, scaler, ridge)
pipe.fit(X_train, y_train)
y_pred_ridge = pipe.predict(X_test)
print("Ridge: ",r2_score(y_test, y_pred_ridge))

pickle.dump(pipe, open('LassoModel.pkl', 'wb'))