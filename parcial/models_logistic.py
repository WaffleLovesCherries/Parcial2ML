from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from numpy import mean

from pickle import load, dump
from multiprocessing import cpu_count

cores = cpu_count() - 1

with open('DataModels.pkl', 'rb') as f:
    X_train, X_bal, _ = load(f)
    y_train, y_bal, _ = load(f)

print( y_train.isnull().sum() )

X_train = X_train.copy()
y_train = y_train.copy()

X_bal = X_bal.copy()
y_bal = y_bal.copy()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=37)
pipeline = Pipeline( [ ( 'scaler', StandardScaler() ), ( 'classifier', LogisticRegression( solver = 'liblinear', max_iter = 500 )) ] )

def target_base( C ):

    param_grid = {
        'classifier__C': [ C ],
        'classifier__penalty': [ 'l1', 'l2' ]
    }

    gs = GridSearchCV( pipeline, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=cores)
    gs.fit(X_train, y_train)
    return mean(gs.cv_results_['mean_test_score'])

pbounds = {'C': ( 0.01, 10 )}

opt = BayesianOptimization(
    f=target_base,
    pbounds=pbounds,
    random_state=37,
    verbose=2
)

opt.maximize( n_iter = 20, init_points = 2 )

param_grid = {
    'classifier__C': [ int( opt.max['params']['C'] ) ],
    'classifier__penalty': [ 'l1', 'l2' ]
}

lr_gs = GridSearchCV( pipeline, param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=2, n_jobs=cores)
lr_gs.fit(X_train, y_train)

def target_ada( C ):
    param_grid = {
        'classifier__C': [ C ],
        'classifier__penalty': [ 'l1', 'l2' ]
    }
    gs = GridSearchCV( pipeline, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=cores)
    gs.fit(X_bal, y_bal)
    return mean(gs.cv_results_['mean_test_score'])

pbounds = {'C': ( 0.01, 10 )}

opt = BayesianOptimization(
    f=target_ada,
    pbounds=pbounds,
    random_state=37,
    verbose=2
)

opt.maximize( n_iter = 20, init_points = 2 )

param_grid = {
    'classifier__C': [ int( opt.max['params']['C'] ) ],
    'classifier__penalty': [ 'l1', 'l2' ]
}

lr_ada_gs = GridSearchCV( pipeline, param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=2, n_jobs=cores)
lr_ada_gs.fit(X_bal, y_bal)

with open('LogisticModel.pkl', 'wb') as f:
    dump(lr_ada_gs, f)
    dump(lr_ada_gs, f)
