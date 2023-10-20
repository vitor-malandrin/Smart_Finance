from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

def tune_linear_regression(train_x, train_y):
    model = LinearRegression()
    parameters = {
        'fit_intercept': [True, False],
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='r2')
    grid_search.fit(train_x, train_y)

    tuned_parameters = grid_search.best_params_
    tuned_model = LinearRegression(**tuned_parameters)
    tuned_model.fit(train_x, train_y)

    return tuned_model

def tune_random_forest(train_x, train_y):
    model = RandomForestRegressor()
    parameters = {
        'max_depth': [None, 10, 20],
        'n_estimators': [50, 100, 150],
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='r2')
    grid_search.fit(train_x, train_y)

    tuned_parameters = grid_search.best_params_
    tuned_model = RandomForestRegressor(**tuned_parameters)
    tuned_model.fit(train_x, train_y)

    return tuned_model

def tune_svr(train_x, train_y):
    model = SVR()
    parameters = {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5],
        'kernel': ['linear', 'poly', 'rbf'],
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(train_x, train_y)

    tuned_parameters = grid_search.best_params_
    tuned_model = SVR(**tuned_parameters)
    tuned_model.fit(train_x, train_y)

    return tuned_model

def tune_gradient_boost(train_x, train_y):
    model = GradientBoostingRegressor()
    parameters = {
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 150],
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='r2')
    grid_search.fit(train_x, train_y)

    tuned_parameters = grid_search.best_params_
    tuned_model = GradientBoostingRegressor(**tuned_parameters)
    tuned_model.fit(train_x, train_y)

    return tuned_model

def tune_ridge(train_x, train_y):
    model = Ridge()
    parameters = {
        'alpha': [0.1, 1, 10],
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='r2')
    grid_search.fit(train_x, train_y)

    tuned_parameters = grid_search.best_params_
    tuned_model = Ridge(**tuned_parameters)
    tuned_model.fit(train_x, train_y)

    return tuned_model

def tune_lasso(train_x, train_y):
    model = Lasso()
    parameters = {
        'alpha': [0.1, 1, 10],
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='r2')
    grid_search.fit(train_x, train_y)

    tuned_parameters = grid_search.best_params_
    tuned_model = Lasso(**tuned_parameters)
    tuned_model.fit(train_x, train_y)

    return tuned_model
