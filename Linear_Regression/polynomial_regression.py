from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

train_data = pd.read_csv("data.csv")
X = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y'].values

poly_feat = PolynomialFeatures(degree= 4)
X_poly = poly_feat.fit_transform(X)

poly_model = None

poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)