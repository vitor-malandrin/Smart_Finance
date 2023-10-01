import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

def linear_regression_test():
    global linear_regression_prediction

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(train_x, train_y)
    linear_regression_prediction = linear_regression_model.predict(test_x)

    print(f'Linear Regression score: {r2_score(test_y, linear_regression_prediction):.6%}')

def random_forest_test():
    global random_forest_prediction

    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(train_x, train_y)
    random_forest_prediction = random_forest_model.predict(test_x)

    print(f'Random Forest score: {r2_score(test_y, random_forest_prediction):.6%}')

def support_vector_test():
    global svr_prediction

    svr_model = SVR()
    svr_model.fit(train_x, train_y)
    svr_prediction = svr_model.predict(test_x)

    print(f'SVR score: {r2_score(test_y, svr_prediction):.6%}')

def gradient_boost_test():
    global gradient_boost_prediction

    gradient_boost_model = GradientBoostingRegressor()
    gradient_boost_model.fit(train_x, train_y)
    gradient_boost_prediction = gradient_boost_model.predict(test_x)

    print(f'Gradient Boost score: {r2_score(test_y, gradient_boost_prediction):.6%}')

def ridge_test():
    global ridge_prediction

    ridge_model = Ridge()
    ridge_model.fit(train_x, train_y)
    ridge_prediction = ridge_model.predict(test_x)

    print(f'Ridge score: {r2_score(test_y, ridge_prediction):.6%}')

def lasso_test():
    global lasso_prediction

    lasso_model = Lasso(max_iter=10000)
    lasso_model.fit(train_x, train_y)
    lasso_prediction = lasso_model.predict(test_x)

    print(f'Lasso score: {r2_score(test_y, lasso_prediction):.6%}')

def test_all_models():
    linear_regression_test()
    random_forest_test()
    support_vector_test()
    gradient_boost_test()
    ridge_test()
    lasso_test()

dataframe = pd.read_csv(r'crypto data\Binance_BTCBRL_d.csv')

x = dataframe[['Open', 'High', 'Low']]
y = dataframe['Close']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

test_all_models()
print()

aux_dataframe = pd.DataFrame()
aux_dataframe['Open'] = test_x['Open']
aux_dataframe['High'] = test_x['High']
aux_dataframe['Low'] = test_x['Low']
aux_dataframe['test_y (Close)'] = test_y
aux_dataframe['Linear Regression'] = linear_regression_prediction
aux_dataframe['Random Forest'] = random_forest_prediction
aux_dataframe['SVR'] = svr_prediction
aux_dataframe['Gradient Boost'] = gradient_boost_prediction
aux_dataframe['Ridge'] = ridge_prediction
aux_dataframe['Lasso'] = lasso_prediction

print(aux_dataframe)

# SVR horrível
# random forest e gradient são os dois piores
# linear, ridge e lasso praticamente idênticas, os melhores
