from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def linear_regression_test(train_x, test_x, train_y, test_y):
    model = LinearRegression()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    print(f'Linear Regression score: {r2_score(test_y, prediction):.6%}')
    return prediction

def random_forest_test(train_x, test_x, train_y, test_y):
    model = RandomForestRegressor()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    print(f'Random Forest score: {r2_score(test_y, prediction):.6%}')
    return prediction

def support_vector_test(train_x, test_x, train_y, test_y):
    model = SVR()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    print(f'SVR score: {r2_score(test_y, prediction):.6%}')
    return prediction

def gradient_boost_test(train_x, test_x, train_y, test_y):
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    print(f'Gradient Boost score: {r2_score(test_y, prediction):.6%}')
    return prediction

def ridge_test(train_x, test_x, train_y, test_y):
    model = Ridge()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    print(f'Ridge score: {r2_score(test_y, prediction):.6%}')
    return prediction

def lasso_test(train_x, test_x, train_y, test_y):
    model = Lasso(max_iter=10000)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    print(f'Lasso score: {r2_score(test_y, prediction):.6%}')
    return prediction
