import pandas as pd
from sklearn.linear_model import LinearRegression

bmi_life_data = pd.read_csv("bmi_life_expectation.csv")

bmi_life_model = LinearRegression()

X = bmi_life_data[['BMI']]
Y = bmi_life_data[['Life expectancy']]

bmi_life_model.fit(X, Y)
laos_life_exp = bmi_life_model.predict([[22]])

print(laos_life_exp)