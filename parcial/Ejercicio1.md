# Breast Cancer

Como se mencionó en el trabajo anterior, es preferible usar la métrica recall, puesto que minimiza el número de tumores maligos predichos, por lo que es menos probable que catalogue un tumor maligno como benigno y ponga la vida de alguien en riesgo por falta de detección


```python
from cleaning import reduce_vif
```

Importación de la base de datos


```python
from sklearn.datasets import load_breast_cancer
from pandas import DataFrame
from numpy import array

data = load_breast_cancer()

X = DataFrame( data = data.data, columns = data.feature_names )
y = array( data.target )
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area error</th>
      <th>compactness error</th>
      <th>concavity error</th>
      <th>concave points error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>153.40</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>74.08</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
    </tr>
    <tr>
      <th>2</th>
      <td>94.03</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.23</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94.44</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_clean = reduce_vif( X )
```

    Dropped col worst texture with vif 63306.17203588469
    Dropped col worst symmetry with vif 63220.51620336962
    Dropped col worst smoothness with vif 63065.70576575069
    Dropped col worst radius with vif 61649.81043724271
    Dropped col worst perimeter with vif 54345.98850641613
    Dropped col worst fractal dimension with vif 50272.811021865375
    Dropped col worst concavity with vif 50212.61746969545
    Dropped col worst concave points with vif 50206.924195754895
    Dropped col worst compactness with vif 50074.052594314235
    Dropped col worst area with vif 45851.514134748824
    Dropped col texture error with vif 45050.18204804375
    Dropped col symmetry error with vif 45006.932477719776
    Dropped col smoothness error with vif 44889.116160428784
    Dropped col radius error with vif 44442.27282332614
    Dropped col perimeter error with vif 35015.04006468088
    Dropped col mean texture with vif 32694.04098810877
    Dropped col mean symmetry with vif 32659.280116484188
    Dropped col mean smoothness with vif 32529.52309425759
    Dropped col mean radius with vif 31807.594519378392
    Dropped col mean perimeter with vif 542.8530090373666
    Dropped col mean fractal dimension with vif 83.0518453152788
    Dropped col mean concavity with vif 79.16240270763147
    Dropped col mean concave points with vif 24.622927588745615
    Dropped col mean compactness with vif 18.149793554937688
    Dropped col mean area with vif 13.509553041668504
    Dropped col fractal dimension error with vif 13.402388859380014
    


```python
X_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area error</th>
      <th>compactness error</th>
      <th>concavity error</th>
      <th>concave points error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>153.40</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>74.08</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
    </tr>
    <tr>
      <th>2</th>
      <td>94.03</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.23</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94.44</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>564</th>
      <td>158.70</td>
      <td>0.02891</td>
      <td>0.05198</td>
      <td>0.02454</td>
    </tr>
    <tr>
      <th>565</th>
      <td>99.04</td>
      <td>0.02423</td>
      <td>0.03950</td>
      <td>0.01678</td>
    </tr>
    <tr>
      <th>566</th>
      <td>48.55</td>
      <td>0.03731</td>
      <td>0.04730</td>
      <td>0.01557</td>
    </tr>
    <tr>
      <th>567</th>
      <td>86.22</td>
      <td>0.06158</td>
      <td>0.07117</td>
      <td>0.01664</td>
    </tr>
    <tr>
      <th>568</th>
      <td>19.15</td>
      <td>0.00466</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 4 columns</p>
</div>



División de set de entrenamiento y prueba


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X_clean, y )
```

## Clasificación Bayesiana

Versión manual


```python
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform( X_clean )

scores = cross_val_score(BernoulliNB(), X_scaled, y, cv=10, scoring='recall')

scores.mean()
```




    0.8234920634920636



Versión con Pipeline


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', BernoulliNB())
])

scores = cross_val_score(pipeline, X_clean, y, cv=10, scoring='recall')

scores.mean()
```




    0.8262698412698413



La diferencia entre los scores se puede deber a la fuga de datos generados por el escalamiento de datos previo a la generación del modelo.

## Arboles de decisión

Versión manual


```python
from sklearn.tree import DecisionTreeClassifier

results = DataFrame( columns = ['Depth', 'recall'] )
for i in range( 4 , 9 ):
    scores = cross_val_score(DecisionTreeClassifier( max_depth=i , random_state=1 ), X_scaled, y, cv=10, scoring='recall')
    results.loc[ len(results) ] = { 'Depth': i, 'recall': scores.mean() }

results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0.924286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0.913175</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0.888175</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>0.874286</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>0.871508</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import GridSearchCV

classifier = DecisionTreeClassifier(random_state=1)

param_grid = {
    'max_depth': [ i for i in range( 4 , 9 ) ]
}

grid_search = GridSearchCV(classifier, param_grid, cv=10, scoring='recall')
grid_search.fit(X_clean, y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Parámetros:", best_params)
print("Recall:", best_score)
```

    Parámetros: {'max_depth': 4}
    Recall: 0.9242857142857142
    

## Random Forest

Versión manual


```python
from sklearn.ensemble  import RandomForestClassifier

results = DataFrame( columns = ['Depth', 'Trees', 'recall'] )
for i in range( 4 , 9 ):
    for j in range( 10 ):
        scores = cross_val_score(RandomForestClassifier( max_depth=i, n_estimators=(j + 1)*5, random_state=1 ), X_scaled, y, cv=10, scoring='recall')
        results.loc[ len(results) ] = { 'Depth': i, 'Trees': (j + 1)*5, 'recall': scores.mean() }

results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>Trees</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>5</td>
      <td>0.938175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>10</td>
      <td>0.943968</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>15</td>
      <td>0.946746</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>20</td>
      <td>0.946746</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>25</td>
      <td>0.943968</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.max()['recall']
```




    0.9467460317460317




```python
classifier = RandomForestClassifier(random_state=1)

param_grid = {
    'max_depth': [ i for i in range( 4, 9 ) ],
    'n_estimators': [ (i + 1)*5 for i in range( 10 ) ]
}

grid_search = GridSearchCV(classifier, param_grid, cv=10, scoring='recall')
grid_search.fit(X_clean, y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Parámetros:", best_params)
print("Recall:", best_score)
```

    Parámetros: {'max_depth': 4, 'n_estimators': 15}
    Recall: 0.9467460317460317
    

## XGBoost


```python
from xgboost  import XGBClassifier

results = DataFrame( columns = ['Depth', 'Trees', 'L. rate', 'recall'] )
for i in range( 4 , 9 ):
    for j in range( 10 ):
        for k in range( 5 ):
            scores = cross_val_score(XGBClassifier( max_depth=i, n_estimators=(j + 1)*5, learning_rate = 10**(-k-1), random_state=1 ), X_scaled, y, cv=10, scoring='recall', n_jobs = -1)
            results.loc[ len(results) ] = { 'Depth': i, 'Trees': (j + 1)*5, 'L. rate': 10**(-k-1), 'recall': scores.mean() }

results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>Trees</th>
      <th>L. rate</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>5</td>
      <td>0.10000</td>
      <td>0.949524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>0.01000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>5</td>
      <td>0.00100</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>0.00010</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0.00001</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.max()['recall']
```




    1.0




```python
param_grid = {
    'max_depth': [ i for i in range( 4, 9 ) ],
    'n_estimators': [ (i + 1)*5 for i in range( 10 ) ],
    'learning_rate': [ 10**(-k-1) for k in range( 5 ) ]
}

grid_search = GridSearchCV(XGBClassifier(random_state=1), param_grid, cv=10, scoring='recall')
grid_search.fit(X_clean, y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Parámetros:", best_params)
print("Recall:", best_score)
```

    Parámetros: {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 5}
    Recall: 1.0
    

Los resultados 


```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

models = [
    ("BernoulliNB", BernoulliNB(), {}),
    ("DecisionTree", DecisionTreeClassifier(max_depth=4), {}),
    ("RandomForest", RandomForestClassifier(max_depth=4, n_estimators=15), {}),
    ("XGBoost", XGBClassifier(learning_rate=0.01, max_depth=4, n_estimators=5), {})
]

results = []

for name, model, params in models:
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results.append([name, precision, recall, f1, auc])

results_df = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1 Score", "AUC"])
results_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BernoulliNB</td>
      <td>0.750000</td>
      <td>0.984076</td>
      <td>0.851240</td>
      <td>0.858352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTree</td>
      <td>0.758270</td>
      <td>0.949045</td>
      <td>0.842999</td>
      <td>0.807749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForest</td>
      <td>0.756039</td>
      <td>0.996815</td>
      <td>0.859890</td>
      <td>0.863273</td>
    </tr>
    <tr>
      <th>3</th>
      <td>XGBoost</td>
      <td>0.612086</td>
      <td>1.000000</td>
      <td>0.759371</td>
      <td>0.792401</td>
    </tr>
  </tbody>
</table>
</div>



Por los resultados, podemos ver que el modelo que maximiza el recall, es un modelo XGBoost con profundidad 4, $\nu = 0.01$ y 5 estimadores. Aún así, es el que peor desempeño obtuvo en otras métricas.

# Boston Housing


```python
import mglearn
import warnings
warnings.filterwarnings("ignore")

X, y = mglearn.datasets.load_extended_boston()
X = pd.DataFrame(X)
```


```python
X_clean = reduce_vif( X )
```

    Dropped col 103.0 with vif inf
    Dropped col 102.0 with vif inf
    Dropped col 101.0 with vif inf
    Dropped col 100.0 with vif inf
    Dropped col 99.0 with vif inf
    Dropped col 98.0 with vif inf
    Dropped col 97.0 with vif inf
    Dropped col 96.0 with vif inf
    Dropped col 95.0 with vif inf
    Dropped col 94.0 with vif inf
    Dropped col 93.0 with vif inf
    Dropped col 92.0 with vif inf
    Dropped col 91.0 with vif inf
    Dropped col 90.0 with vif inf
    Dropped col 89.0 with vif inf
    Dropped col 88.0 with vif inf
    Dropped col 87.0 with vif inf
    Dropped col 86.0 with vif inf
    Dropped col 85.0 with vif inf
    Dropped col 84.0 with vif inf
    Dropped col 83.0 with vif inf
    Dropped col 82.0 with vif inf
    Dropped col 81.0 with vif inf
    Dropped col 80.0 with vif inf
    Dropped col 79.0 with vif inf
    Dropped col 78.0 with vif inf
    Dropped col 77.0 with vif inf
    Dropped col 76.0 with vif inf
    Dropped col 75.0 with vif inf
    Dropped col 74.0 with vif inf
    Dropped col 73.0 with vif inf
    Dropped col 72.0 with vif inf
    Dropped col 71.0 with vif inf
    Dropped col 70.0 with vif inf
    Dropped col 69.0 with vif inf
    Dropped col 68.0 with vif inf
    Dropped col 67.0 with vif inf
    Dropped col 66.0 with vif inf
    Dropped col 65.0 with vif inf
    Dropped col 64.0 with vif inf
    Dropped col 63.0 with vif inf
    Dropped col 62.0 with vif inf
    Dropped col 61.0 with vif inf
    Dropped col 60.0 with vif inf
    Dropped col 59.0 with vif inf
    Dropped col 58.0 with vif inf
    Dropped col 57.0 with vif inf
    Dropped col 56.0 with vif inf
    Dropped col 55.0 with vif inf
    Dropped col 54.0 with vif inf
    Dropped col 53.0 with vif inf
    Dropped col 52.0 with vif inf
    Dropped col 51.0 with vif inf
    Dropped col 50.0 with vif inf
    Dropped col 49.0 with vif inf
    Dropped col 48.0 with vif 1346281.6963565508
    Dropped col 47.0 with vif 1320337.0328633178
    Dropped col 46.0 with vif 1301422.0813645376
    Dropped col 45.0 with vif 1263484.9108551014
    Dropped col 44.0 with vif 1156525.8235163095
    Dropped col 43.0 with vif 1156420.4311001496
    Dropped col 42.0 with vif 1130972.6590758632
    Dropped col 41.0 with vif 1112118.8289185707
    Dropped col 40.0 with vif 1111983.4174159167
    Dropped col 39.0 with vif 1103841.2032959189
    Dropped col 38.0 with vif 1095373.3319310762
    Dropped col 37.0 with vif 1079473.9966043278
    Dropped col 36.0 with vif 1079418.1975485373
    Dropped col 35.0 with vif 1078102.7615766579
    Dropped col 34.0 with vif 1065479.5086273584
    Dropped col 33.0 with vif 954935.5716944332
    Dropped col 32.0 with vif 932086.2637821351
    Dropped col 31.0 with vif 929393.7230950865
    Dropped col 30.0 with vif 927618.1668583998
    Dropped col 29.0 with vif 907195.1078493975
    Dropped col 28.0 with vif 905889.8090825254
    Dropped col 27.0 with vif 896909.5148233331
    Dropped col 26.0 with vif 877374.2013817845
    Dropped col 25.0 with vif 876296.4010351859
    Dropped col 24.0 with vif 874177.7500446256
    Dropped col 23.0 with vif 874176.4876008927
    Dropped col 22.0 with vif 632015.2581415928
    Dropped col 21.0 with vif 6644.49434478578
    Dropped col 20.0 with vif 6214.28009225289
    Dropped col 19.0 with vif 5930.265237708079
    Dropped col 18.0 with vif 5865.861819563112
    Dropped col 17.0 with vif 5865.853152114421
    Dropped col 16.0 with vif 5782.276045524795
    Dropped col 15.0 with vif 5778.311737636222
    Dropped col 14.0 with vif 24.714436314764924
    Dropped col 13.0 with vif 24.558573695096555
    Dropped col 12.0 with vif 24.55271748545112
    Dropped col 11.0 with vif 24.55077296778318
    Dropped col 10.0 with vif 24.541531486873993
    Dropped col 9.0 with vif 24.365332542910693
    Dropped col 8.0 with vif 14.003149917329047
    Dropped col 7.0 with vif 13.919131411917435
    Dropped col 6.0 with vif 13.893467219259236
    


```python
X_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.18</td>
      <td>0.067815</td>
      <td>0.0</td>
      <td>0.314815</td>
      <td>0.577505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000236</td>
      <td>0.00</td>
      <td>0.242302</td>
      <td>0.0</td>
      <td>0.172840</td>
      <td>0.547998</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000236</td>
      <td>0.00</td>
      <td>0.242302</td>
      <td>0.0</td>
      <td>0.172840</td>
      <td>0.694386</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000293</td>
      <td>0.00</td>
      <td>0.063050</td>
      <td>0.0</td>
      <td>0.150206</td>
      <td>0.658555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000705</td>
      <td>0.00</td>
      <td>0.063050</td>
      <td>0.0</td>
      <td>0.150206</td>
      <td>0.687105</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>501</th>
      <td>0.000633</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.386831</td>
      <td>0.580954</td>
    </tr>
    <tr>
      <th>502</th>
      <td>0.000438</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.386831</td>
      <td>0.490324</td>
    </tr>
    <tr>
      <th>503</th>
      <td>0.000612</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.386831</td>
      <td>0.654340</td>
    </tr>
    <tr>
      <th>504</th>
      <td>0.001161</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.386831</td>
      <td>0.619467</td>
    </tr>
    <tr>
      <th>505</th>
      <td>0.000462</td>
      <td>0.00</td>
      <td>0.420455</td>
      <td>0.0</td>
      <td>0.386831</td>
      <td>0.473079</td>
    </tr>
  </tbody>
</table>
<p>506 rows × 6 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split( X_clean, y )
```

Dado que esta regresión no requiere hiperparámetros ni escalado de datos, bastará con la generación del modelo.


```python
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score

model = BayesianRidge()

model.fit( X_train, y_train )
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", mse**(1/2))
print("R^2:", r2)
```

    RMSE: 5.72422243657148
    R^2: 0.5820983222977625
    

## Árbol de decisión


```python
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Assuming X_scaled and y are your features and labels

results = pd.DataFrame(columns=['Depth', 'MSE', 'R^2'])

for i in range(4, 9):
    regressor = DecisionTreeRegressor(max_depth=i, random_state=1)
    scores_mse = -cross_val_score(regressor, X_clean, y, cv=10, scoring='neg_mean_squared_error')
    scores_r2 = cross_val_score(regressor, X_clean, y, cv=10, scoring='r2')
    
    results.loc[len(results)] = {'Depth': i, 'MSE': scores_mse.mean(), 'R^2': scores_r2.mean()}

results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>MSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>47.881421</td>
      <td>-1.104367</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>58.197278</td>
      <td>-2.476104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>70.858490</td>
      <td>-3.419201</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>75.141397</td>
      <td>-4.801361</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>79.066248</td>
      <td>-4.883838</td>
    </tr>
  </tbody>
</table>
</div>



La explicación del $R^2$ se puede encontrar aquí: https://towardsdatascience.com/explaining-negative-r-squared-17894ca26321, pero en escencia, se dan valores negativos cuando el modelo predice la variable respuesta peor que un modelo constante que retorna siempre la media.


```python
regressor = DecisionTreeRegressor(random_state=1)

param_grid = {
    'max_depth': [i for i in range(4, 9)]
}

grid_search = GridSearchCV(regressor, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_clean, y)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Take the negative to get MSE

print("Best Parameters:", best_params)
print("Best MSE:", best_score)
```

    Best Parameters: {'max_depth': 4}
    Best MSE: 47.8814214874713
    


```python
regressor = DecisionTreeRegressor(random_state=1)

param_grid = {
    'max_depth': [i for i in range(4, 9)]
}

grid_search = GridSearchCV(regressor, param_grid, cv=10, scoring='r2')
grid_search.fit(X_clean, y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_  # Take the negative to get MSE

print("Best Parameters:", best_params)
print("Best R^2:", best_score)
```

    Best Parameters: {'max_depth': 4}
    Best R^2: -1.104367077003224
    

## Random Forest


```python
from sklearn.ensemble import RandomForestRegressor

results = pd.DataFrame(columns=['Depth', 'Trees', 'MSE', 'R^2'])

for i in range(4, 9):
    for j in range(10):
        regressor = RandomForestRegressor(max_depth=i, n_estimators=(j + 1) * 5, random_state=1)
        scores_mse = -cross_val_score(regressor, X_clean, y, cv=10, scoring='neg_mean_squared_error')
        scores_r2 = cross_val_score(regressor, X_clean, y, cv=10, scoring='r2')
        
        results.loc[len(results)] = {'Depth': i, 'Trees': (j + 1) * 5, 'MSE': scores_mse.mean(), 'R^2': scores_r2.mean()}

results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>Trees</th>
      <th>MSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>5</td>
      <td>34.766108</td>
      <td>0.010018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>10</td>
      <td>32.571282</td>
      <td>0.082020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>15</td>
      <td>32.494404</td>
      <td>0.072480</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>20</td>
      <td>31.867860</td>
      <td>0.075231</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>25</td>
      <td>31.488796</td>
      <td>0.076304</td>
    </tr>
  </tbody>
</table>
</div>




```python
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>Trees</th>
      <th>MSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>5</td>
      <td>34.766108</td>
      <td>0.010018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>10</td>
      <td>32.571282</td>
      <td>0.082020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>15</td>
      <td>32.494404</td>
      <td>0.072480</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>20</td>
      <td>31.867860</td>
      <td>0.075231</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>25</td>
      <td>31.488796</td>
      <td>0.076304</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>30</td>
      <td>32.102933</td>
      <td>0.045308</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>35</td>
      <td>31.615452</td>
      <td>0.076428</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>40</td>
      <td>31.226574</td>
      <td>0.071318</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>45</td>
      <td>31.468311</td>
      <td>0.048010</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>50</td>
      <td>31.315379</td>
      <td>0.061626</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5</td>
      <td>5</td>
      <td>38.214430</td>
      <td>-0.276892</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>10</td>
      <td>34.982822</td>
      <td>-0.157419</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>15</td>
      <td>34.915417</td>
      <td>-0.216604</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>20</td>
      <td>34.029162</td>
      <td>-0.166275</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>25</td>
      <td>33.438062</td>
      <td>-0.152460</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>30</td>
      <td>33.795906</td>
      <td>-0.160293</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>35</td>
      <td>33.141826</td>
      <td>-0.088163</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>40</td>
      <td>32.622444</td>
      <td>-0.096594</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5</td>
      <td>45</td>
      <td>32.818188</td>
      <td>-0.131144</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5</td>
      <td>50</td>
      <td>32.564593</td>
      <td>-0.096418</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6</td>
      <td>5</td>
      <td>38.544257</td>
      <td>-0.207476</td>
    </tr>
    <tr>
      <th>21</th>
      <td>6</td>
      <td>10</td>
      <td>37.427726</td>
      <td>-0.375445</td>
    </tr>
    <tr>
      <th>22</th>
      <td>6</td>
      <td>15</td>
      <td>36.733992</td>
      <td>-0.430453</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6</td>
      <td>20</td>
      <td>34.893727</td>
      <td>-0.279512</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6</td>
      <td>25</td>
      <td>33.755048</td>
      <td>-0.226830</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6</td>
      <td>30</td>
      <td>34.466766</td>
      <td>-0.286255</td>
    </tr>
    <tr>
      <th>26</th>
      <td>6</td>
      <td>35</td>
      <td>33.724054</td>
      <td>-0.204135</td>
    </tr>
    <tr>
      <th>27</th>
      <td>6</td>
      <td>40</td>
      <td>33.136167</td>
      <td>-0.206413</td>
    </tr>
    <tr>
      <th>28</th>
      <td>6</td>
      <td>45</td>
      <td>33.226287</td>
      <td>-0.193386</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6</td>
      <td>50</td>
      <td>32.822166</td>
      <td>-0.143083</td>
    </tr>
    <tr>
      <th>30</th>
      <td>7</td>
      <td>5</td>
      <td>37.219519</td>
      <td>-0.242167</td>
    </tr>
    <tr>
      <th>31</th>
      <td>7</td>
      <td>10</td>
      <td>36.977551</td>
      <td>-0.392013</td>
    </tr>
    <tr>
      <th>32</th>
      <td>7</td>
      <td>15</td>
      <td>35.788518</td>
      <td>-0.342091</td>
    </tr>
    <tr>
      <th>33</th>
      <td>7</td>
      <td>20</td>
      <td>34.432608</td>
      <td>-0.272858</td>
    </tr>
    <tr>
      <th>34</th>
      <td>7</td>
      <td>25</td>
      <td>34.024405</td>
      <td>-0.249124</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7</td>
      <td>30</td>
      <td>34.303980</td>
      <td>-0.264675</td>
    </tr>
    <tr>
      <th>36</th>
      <td>7</td>
      <td>35</td>
      <td>33.728431</td>
      <td>-0.187957</td>
    </tr>
    <tr>
      <th>37</th>
      <td>7</td>
      <td>40</td>
      <td>32.987389</td>
      <td>-0.146079</td>
    </tr>
    <tr>
      <th>38</th>
      <td>7</td>
      <td>45</td>
      <td>33.260962</td>
      <td>-0.175985</td>
    </tr>
    <tr>
      <th>39</th>
      <td>7</td>
      <td>50</td>
      <td>32.870736</td>
      <td>-0.127700</td>
    </tr>
    <tr>
      <th>40</th>
      <td>8</td>
      <td>5</td>
      <td>37.688317</td>
      <td>-0.253550</td>
    </tr>
    <tr>
      <th>41</th>
      <td>8</td>
      <td>10</td>
      <td>35.786831</td>
      <td>-0.128893</td>
    </tr>
    <tr>
      <th>42</th>
      <td>8</td>
      <td>15</td>
      <td>35.850802</td>
      <td>-0.296612</td>
    </tr>
    <tr>
      <th>43</th>
      <td>8</td>
      <td>20</td>
      <td>34.671280</td>
      <td>-0.238545</td>
    </tr>
    <tr>
      <th>44</th>
      <td>8</td>
      <td>25</td>
      <td>34.391073</td>
      <td>-0.227171</td>
    </tr>
    <tr>
      <th>45</th>
      <td>8</td>
      <td>30</td>
      <td>34.762659</td>
      <td>-0.268420</td>
    </tr>
    <tr>
      <th>46</th>
      <td>8</td>
      <td>35</td>
      <td>34.144845</td>
      <td>-0.191317</td>
    </tr>
    <tr>
      <th>47</th>
      <td>8</td>
      <td>40</td>
      <td>33.716304</td>
      <td>-0.205125</td>
    </tr>
    <tr>
      <th>48</th>
      <td>8</td>
      <td>45</td>
      <td>33.806146</td>
      <td>-0.189767</td>
    </tr>
    <tr>
      <th>49</th>
      <td>8</td>
      <td>50</td>
      <td>33.441433</td>
      <td>-0.143658</td>
    </tr>
  </tbody>
</table>
</div>



Dado que el $R^2$ es mínimo, se buscará reducir el MSE lo más que se pueda, en este caso es en el índice 7, con 40 árboles y 4 de profundidad máxima.


```python
regressor = RandomForestRegressor(random_state=1)

param_grid = {
    'max_depth': [i for i in range(4, 9)],
    'n_estimators': [(i + 1) * 5 for i in range(10)]
}

grid_search = GridSearchCV(regressor, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_clean, y)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print("Best Parameters:", best_params)
print("Best MSE:", best_score)
```

    Best Parameters: {'max_depth': 4, 'n_estimators': 40}
    Best MSE: 31.226574256687353
    

## XGBoost


```python
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd

results = pd.DataFrame(columns=['Depth', 'Trees', 'Learning Rate', 'MSE', 'R^2'])

for i in range(4, 9):
    for j in range(10):
        for k in range(5):
            regressor = XGBRegressor(max_depth=i, n_estimators=(j + 1) * 5, learning_rate=10 ** (-k - 1), random_state=1)
            scores_mse = -cross_val_score(regressor, X_clean, y, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
            scores_r2 = cross_val_score(regressor, X_clean, y, cv=10, scoring='r2', n_jobs=-1)
            
            results.loc[len(results)] = {'Depth': i, 'Trees': (j + 1) * 5, 'Learning Rate': 10 ** (-k - 1),
                                          'MSE': scores_mse.mean(), 'R^2': scores_r2.mean()}

results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>Trees</th>
      <th>Learning Rate</th>
      <th>MSE</th>
      <th>R^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>5</td>
      <td>0.10000</td>
      <td>53.990609</td>
      <td>-0.299513</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>0.01000</td>
      <td>86.529409</td>
      <td>-1.146103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>5</td>
      <td>0.00100</td>
      <td>91.521877</td>
      <td>-1.271545</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>0.00010</td>
      <td>92.043544</td>
      <td>-1.284624</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0.00001</td>
      <td>92.095905</td>
      <td>-1.285937</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.max()
```




    Depth             8.000000
    Trees            50.000000
    Learning Rate     0.100000
    MSE              92.095905
    R^2               0.267295
    dtype: float64




```python
param_grid = {
    'max_depth': [i for i in range(4, 9)],
    'n_estimators': [(i + 1) * 5 for i in range(10)],
    'learning_rate': [10 ** (-k - 1) for k in range(5)]
}

grid_search = GridSearchCV(XGBRegressor(random_state=1), param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_clean, y)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Take the negative to get the actual MSE

print("Best Parameters:", best_params)
print("Best MSE:", best_score)
```

    Best Parameters: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 25}
    Best MSE: 32.61838071368801
    

## Resultados finales


```python
from numpy import mean, abs, sqrt

models = [
    ("Bayesian Ridge", BayesianRidge()),
    ("Decision Tree", DecisionTreeRegressor(max_depth=4)),
    ("Random Forest", RandomForestRegressor(max_depth=4, n_estimators=40)),
    ("XGBoost", XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=25))
]

results = DataFrame(columns=["Model", "RMSE", "R2", "MAPE"])

def mean_absolute_percentage_error(y_true, y_pred):
    return mean(abs((y_true - y_pred) / y_true)) * 100

results = []
for name, model in models:
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    results.append({"Model": name, "RMSE": rmse, "R2": r2, "MAPE": mape})

DataFrame(results)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>R2</th>
      <th>MAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bayesian Ridge</td>
      <td>5.724222</td>
      <td>0.582098</td>
      <td>21.214606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>5.467618</td>
      <td>0.618726</td>
      <td>15.424934</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>5.246252</td>
      <td>0.648974</td>
      <td>15.361242</td>
    </tr>
    <tr>
      <th>3</th>
      <td>XGBoost</td>
      <td>4.915800</td>
      <td>0.691802</td>
      <td>14.608670</td>
    </tr>
  </tbody>
</table>
</div>



Curiosamente, los problemas vistos en el $R^2$ desaparecen con el conjunto de prueba. Aún así vemos que ningún modelo es ideal para la predicción.
