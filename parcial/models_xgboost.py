from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.base import clone
from bayes_opt import BayesianOptimization
from numpy import mean

from pickle import load, dump
from multiprocessing import cpu_count

cores = cpu_count() - 1

with open('DataML.pkl', 'rb') as f:
    X_train, X_bal, _ = load(f)
    y_train, y_bal, _ = load(f)

print('XGBoost')

X_train = X_train.copy()
y_train = y_train.copy()

X_bal = X_bal.copy()
y_bal = y_bal.copy()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=37)

def target_base( n_estimators, learning_rate ):
    param_grid = {
        'n_estimators': [ int(n_estimators) ],
        'max_depth': [ i + 4 for i in range(5) ],
        'learning_rate': [ learning_rate ]
    }
    gs = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=cores)
    gs.fit(X_train, y_train)
    return mean(gs.cv_results_['mean_test_score'])

pbounds = {'n_estimators': (1, 200), 'learning_rate': (0.0001, 0.09)}

opt = BayesianOptimization(
    f=target_base,
    pbounds=pbounds,
    random_state=37,
    verbose=2
)

opt.maximize( n_iter = 10, init_points = 2 )

param_grid = {
    'n_estimators': [ int( opt.max['params']['n_estimators'] ) ],
    'max_depth': [  i + 4 for i in range(5)  ],
    'learning_rate': [ opt.max['params']['learning_rate'] ]
}

xgb_gs = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=2, n_jobs=cores)
xgb_gs.fit(X_train, y_train)

def target_base_ada( n_estimators, learning_rate ):
    param_grid = {
        'n_estimators': [ int(n_estimators) ],
        'max_depth': [ i + 4 for i in range(5) ],
        'learning_rate': [ learning_rate ]
    }
    gs = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=cores)
    gs.fit(X_bal, y_bal)
    return mean(gs.cv_results_['mean_test_score'])

opt = BayesianOptimization(
    f=target_base_ada,
    pbounds=pbounds,
    random_state=37,
    verbose=2
)

opt.maximize(n_iter=10, init_points=2)

param_grid = {
    'n_estimators': [ int( opt.max['params']['n_estimators'] ) ],
    'max_depth': [  i + 4 for i in range(5)  ],
    'learning_rate': [ opt.max['params']['learning_rate'] ]
}

xgb_ada_gs = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=cv, scoring='roc_auc', verbose=2, n_jobs=cores)
xgb_ada_gs.fit(X_bal, y_bal)

with open('XGBoostModel.pkl', 'wb') as f:
    dump(xgb_gs, f)
    dump(xgb_ada_gs, f)
