# Naive Bayes


```python
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from pickle import load, dump
from multiprocessing import cpu_count

cores = cpu_count() - 1

with open( 'DataML.pkl', 'rb' ) as f:
    X_train, X_bal, _ = load( f )
    y_train, y_bal, _ = load( f )

print('Bayesian')

X_train = X_train.copy()
y_train = y_train.copy()

X_bal = X_bal.copy()
y_bal = y_bal.copy()

nb_pl = Pipeline([
    ( 'scaler', StandardScaler() ),
    ( 'classifier', BernoulliNB() )
])

nb_ada_pl: Pipeline = clone( nb_pl )

nb_pl.fit( X_train, y_train )
nb_ada_pl.fit( X_bal, y_bal )

with open( 'NaiveBayesModel.pkl', 'wb' ) as f:
    dump( nb_pl, f )
    dump( nb_ada_pl, f )

```

# Decision Tree


```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from pickle import load, dump
from multiprocessing import cpu_count

cores = cpu_count() - 1

with open( 'DataML.pkl', 'rb' ) as f:
    X_train, X_bal, _ = load( f )
    y_train, y_bal, _ = load( f )

print('Decision Tree')
X_train = X_train.copy()
y_train = y_train.copy()

X_bal = X_bal.copy()
y_bal = y_bal.copy()

param_grid = {
    'max_depth': [ i + 4 for i in range(5) ]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state = 37 )

dt_gs = GridSearchCV( DecisionTreeClassifier(), param_grid = param_grid, cv = cv, scoring = 'roc_auc', verbose = 2, n_jobs = cores )
dt_ada_gs: GridSearchCV = clone( dt_gs )

dt_gs.fit( X_train, y_train )
dt_ada_gs.fit( X_bal, y_bal )

with open( 'DecisionTreeModel.pkl', 'wb' ) as f:
    dump( dt_gs, f )
    dump( dt_ada_gs, f )

```

# Random Forest


```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from bayes_opt import BayesianOptimization
from numpy import mean

from pickle import load, dump
from multiprocessing import cpu_count

cores = - 1

with open( 'DataML.pkl', 'rb' ) as f:
    X_train, X_bal, _ = load( f )
    y_train, y_bal, _ = load( f )

print('Random Forest')
X_train = X_train.copy()
y_train = y_train.copy()

X_bal = X_bal.copy()
y_bal = y_bal.copy()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state = 37 )

def target_base( n_estimators ):
    param_grid = {
        'max_depth': [ i + 4 for i in range(5) ],
        'n_estimators': [ int(n_estimators) ]
    }
    gs = GridSearchCV( RandomForestClassifier(), param_grid = param_grid, cv = cv, scoring = 'roc_auc', n_jobs = cores )
    gs.fit( X_train, y_train )
    return mean( gs.cv_results_['mean_test_score'] )

pbounds = { 'n_estimators': (100,200) }

opt = BayesianOptimization(
    f=target_base,
    pbounds=pbounds,
    random_state=37,
    verbose=2
)

opt.maximize( n_iter = 6, init_points = 2 )

param_grid = {
    'max_depth': [ i + 4 for i in range(5) ],
    'n_estimators': [ int( opt.max['params']['n_estimators'] ) ]
}

rf_gs = GridSearchCV( RandomForestClassifier(), param_grid = param_grid, cv = cv, scoring = 'roc_auc', verbose = 2, n_jobs = cores )
rf_gs.fit( X_train, y_train )


def target_base_ada( n_estimators ):
    param_grid = {
        'max_depth': [ i + 4 for i in range(5) ],
        'n_estimators': [ int(n_estimators) ]
    }
    gs = GridSearchCV( RandomForestClassifier(), param_grid = param_grid, cv = cv, scoring = 'roc_auc', n_jobs = cores )
    gs.fit( X_bal, y_bal )
    return mean( gs.cv_results_['mean_test_score'] )

pbounds = { 'n_estimators': (100,200) }

opt = BayesianOptimization(
    f=target_base_ada,
    pbounds=pbounds,
    random_state=37,
    verbose=2
)

opt.maximize( n_iter = 6, init_points = 2 )

param_grid = {
    'max_depth': [ i + 4 for i in range(5) ],
    'n_estimators': [ int( opt.max['params']['n_estimators'] ) ]
}

rf_ada_gs = GridSearchCV( RandomForestClassifier(), param_grid = param_grid, cv = cv, scoring = 'roc_auc', verbose = 2, n_jobs = cores )
rf_ada_gs.fit( X_bal, y_bal )

with open( 'RandomForestModel.pkl', 'wb' ) as f:
    dump( rf_gs, f )
    dump( rf_ada_gs, f )
```

# XGBoost


```python
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

```

# k-NN


```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from bayes_opt import BayesianOptimization
from numpy import mean

from pickle import dump, load
from multiprocessing import cpu_count

cores = - 1

with open('DataML.pkl', 'rb') as f:
    X_train, X_bal, _ = load(f)
    y_train, y_bal, _ = load(f)

print('K-Nearest Neighbors')

X_train = X_train.copy()
y_train = y_train.copy()

X_bal = X_bal.copy()
y_bal = y_bal.copy()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=37)

def target_base( n_neighbors ):
    knn = KNeighborsClassifier(n_neighbors=int(n_neighbors))
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=cores)
    return mean(scores)

pbounds = {'n_neighbors': (1, 400)}

opt = BayesianOptimization(
    f=target_base,
    pbounds=pbounds,
    random_state=37,
    verbose=2,
    allow_duplicate_points=True
)

opt.maximize(n_iter=5, init_points=2)

knn = KNeighborsClassifier(n_neighbors=int(opt.max['params']['n_neighbors']))
knn.fit(X_train, y_train)

def target_ada( n_neighbors ):
    knn = KNeighborsClassifier(n_neighbors=int(n_neighbors))
    scores = cross_val_score(knn, X_bal, y_bal, cv=cv, scoring='roc_auc', n_jobs=cores)
    return mean(scores)

pbounds = {'n_neighbors': (1, 400)}

opt = BayesianOptimization(
    f=target_ada,
    pbounds=pbounds,
    random_state=37,
    verbose=2,
    allow_duplicate_points=True
)

opt.maximize(n_iter=5, init_points=2)

knn_ada = KNeighborsClassifier(n_neighbors=int(opt.max['params']['n_neighbors']))
knn_ada.fit(X_bal, y_bal)

with open('KNN_Model.pkl', 'wb') as f:
    dump( knn, f )
    dump( knn_ada, f )
```

# Regresion Logística


```python
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
    dump(lr_gs, f)
    dump(lr_ada_gs, f)

```
