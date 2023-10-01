import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataframe = pd.read_csv(r'crypto data\Binance_BTCBRL_d.csv')

x = dataframe[['Open', 'High', 'Low']]
y = dataframe['Close']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

linear_regression_model = LinearRegression()
random_forest_model = RandomForestRegressor()

linear_regression_model.fit(train_x, train_y)
random_forest_model.fit(train_x, train_y)

linear_regression_prediction = linear_regression_model.predict(test_x)
random_forest_prediction = random_forest_model.predict(test_x)

print(f'Precisão do modelo de Regressão Linear: {r2_score(test_y, linear_regression_prediction):.6%}')
print(f'Precisão do modelo da Árvore de Decisão: {r2_score(test_y, random_forest_prediction):.6%}', '\n')

aux_dataframe = pd.DataFrame()
aux_dataframe['Open'] = test_x['Open']
aux_dataframe['High'] = test_x['High']
aux_dataframe['Low'] = test_x['Low']
aux_dataframe['test_y'] = test_y
aux_dataframe['Linear Regression prediction'] = linear_regression_prediction
aux_dataframe['Random Forest prediction'] = random_forest_prediction

print(aux_dataframe)

# regressão linear superior em praticamente todas as execuções


