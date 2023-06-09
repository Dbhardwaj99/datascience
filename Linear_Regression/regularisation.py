import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


train_data = pd.read_csv("data.csv", header = 0)
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

lasso_reg = Lasso()

lasso_reg.fit(X, y)

reg_coef = lasso_reg.coef_
print(reg_coef)
