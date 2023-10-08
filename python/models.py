from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def linear_regression_test(train_x, test_x, train_y, test_y):
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(train_x, train_y)
    linear_regression_prediction = linear_regression_model.predict(test_x)

    print(f'Linear Regression score: {r2_score(test_y, linear_regression_prediction):.6%}')
    return linear_regression_prediction

def random_forest_test(train_x, test_x, train_y, test_y):
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(train_x, train_y)
    random_forest_prediction = random_forest_model.predict(test_x)

    print(f'Random Forest score: {r2_score(test_y, random_forest_prediction):.6%}')
    return random_forest_prediction

def support_vector_test(train_x, test_x, train_y, test_y):
    support_vector_model = SVR()
    support_vector_model.fit(train_x, train_y)
    support_vector_prediction = support_vector_model.predict(test_x)

    print(f'SVR score: {r2_score(test_y, support_vector_prediction):.6%}')
    return support_vector_prediction

def gradient_boost_test(train_x, test_x, train_y, test_y):
    gradient_boost_model = GradientBoostingRegressor()
    gradient_boost_model.fit(train_x, train_y)
    gradient_boost_prediction = gradient_boost_model.predict(test_x)

    print(f'Gradient Boost score: {r2_score(test_y, gradient_boost_prediction):.6%}')
    return gradient_boost_prediction

def ridge_test(train_x, test_x, train_y, test_y):
    ridge_model = Ridge()
    ridge_model.fit(train_x, train_y)
    ridge_prediction = ridge_model.predict(test_x)

    print(f'Ridge score: {r2_score(test_y, ridge_prediction):.6%}')
    return ridge_prediction

def lasso_test(train_x, test_x, train_y, test_y):
    lasso_model = Lasso(max_iter=10000)
    lasso_model.fit(train_x, train_y)
    lasso_prediction = lasso_model.predict(test_x)

    print(f'Lasso score: {r2_score(test_y, lasso_prediction):.6%}')
    return lasso_prediction
