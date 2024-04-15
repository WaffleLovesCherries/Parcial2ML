Importación de librerias y scripts.


```python
from pickle import load, dump
from pandas import DataFrame, concat
from cleaning import chisq_matrix

with open( 'DataRaw.pkl', 'rb' ) as f:
    DataRaw = load(f)
    DatRawCat = load(f)

with open( 'DataClean.pkl', 'rb' ) as f:
    DataClean = load(f)
    DataCat = load(f)
```


```python
DataRaw
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
      <th>TransactionID</th>
      <th>isFraud</th>
      <th>TransactionDT</th>
      <th>TransactionAmt</th>
      <th>card1</th>
      <th>card2</th>
      <th>card3</th>
      <th>card5</th>
      <th>addr1</th>
      <th>addr2</th>
      <th>...</th>
      <th>id_17</th>
      <th>id_18</th>
      <th>id_19</th>
      <th>id_20</th>
      <th>id_21</th>
      <th>id_22</th>
      <th>id_24</th>
      <th>id_25</th>
      <th>id_26</th>
      <th>id_32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2987000</td>
      <td>0</td>
      <td>86400</td>
      <td>68.50</td>
      <td>13926</td>
      <td>362.555488</td>
      <td>150.0</td>
      <td>142.0</td>
      <td>315.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2987001</td>
      <td>0</td>
      <td>86401</td>
      <td>29.00</td>
      <td>2755</td>
      <td>404.000000</td>
      <td>150.0</td>
      <td>102.0</td>
      <td>325.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2987002</td>
      <td>0</td>
      <td>86469</td>
      <td>59.00</td>
      <td>4663</td>
      <td>490.000000</td>
      <td>150.0</td>
      <td>166.0</td>
      <td>330.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2987003</td>
      <td>0</td>
      <td>86499</td>
      <td>50.00</td>
      <td>18132</td>
      <td>567.000000</td>
      <td>150.0</td>
      <td>117.0</td>
      <td>476.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2987004</td>
      <td>0</td>
      <td>86506</td>
      <td>50.00</td>
      <td>4497</td>
      <td>514.000000</td>
      <td>150.0</td>
      <td>102.0</td>
      <td>420.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>166.000000</td>
      <td>14.237337</td>
      <td>542.000000</td>
      <td>144.000000</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>32.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>590535</th>
      <td>3577535</td>
      <td>0</td>
      <td>15811047</td>
      <td>49.00</td>
      <td>6550</td>
      <td>362.555488</td>
      <td>150.0</td>
      <td>226.0</td>
      <td>272.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>590536</th>
      <td>3577536</td>
      <td>0</td>
      <td>15811049</td>
      <td>39.50</td>
      <td>10444</td>
      <td>225.000000</td>
      <td>150.0</td>
      <td>224.0</td>
      <td>204.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>590537</th>
      <td>3577537</td>
      <td>0</td>
      <td>15811079</td>
      <td>30.95</td>
      <td>12037</td>
      <td>595.000000</td>
      <td>150.0</td>
      <td>224.0</td>
      <td>231.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>590538</th>
      <td>3577538</td>
      <td>0</td>
      <td>15811088</td>
      <td>117.00</td>
      <td>7826</td>
      <td>481.000000</td>
      <td>150.0</td>
      <td>224.0</td>
      <td>387.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
    <tr>
      <th>590539</th>
      <td>3577539</td>
      <td>0</td>
      <td>15811131</td>
      <td>279.95</td>
      <td>15066</td>
      <td>170.000000</td>
      <td>150.0</td>
      <td>102.0</td>
      <td>299.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>189.451377</td>
      <td>14.237337</td>
      <td>353.128174</td>
      <td>403.882666</td>
      <td>368.26982</td>
      <td>16.002708</td>
      <td>12.800927</td>
      <td>329.608924</td>
      <td>149.070308</td>
      <td>26.508597</td>
    </tr>
  </tbody>
</table>
<p>590540 rows × 403 columns</p>
</div>




```python
DataClean
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
      <th>C1</th>
      <th>C3</th>
      <th>C5</th>
      <th>D1</th>
      <th>D3</th>
      <th>D10</th>
      <th>D11</th>
      <th>D12</th>
      <th>D13</th>
      <th>D14</th>
      <th>...</th>
      <th>V104</th>
      <th>V101</th>
      <th>V100</th>
      <th>V10</th>
      <th>TransactionAmt</th>
      <th>D9</th>
      <th>D8</th>
      <th>D7</th>
      <th>D5</th>
      <th>D4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>68.50</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>42.335965</td>
      <td>140.002441</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28.343348</td>
      <td>0.000000</td>
      <td>146.621465</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.463915</td>
      <td>29.00</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>42.335965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28.343348</td>
      <td>0.000000</td>
      <td>315.000000</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>59.00</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>42.335965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>112.0</td>
      <td>0.000000</td>
      <td>84.000000</td>
      <td>146.621465</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.463915</td>
      <td>50.00</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>0.000000</td>
      <td>94.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28.343348</td>
      <td>123.982137</td>
      <td>146.621465</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.463915</td>
      <td>50.00</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>42.335965</td>
      <td>140.002441</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>590535</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>30.000000</td>
      <td>56.000000</td>
      <td>56.000000</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>49.00</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>42.335965</td>
      <td>140.002441</td>
    </tr>
    <tr>
      <th>590536</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28.343348</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>39.50</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>42.335965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>590537</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>28.343348</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>30.95</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>42.335965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>590538</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>22.000000</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>117.00</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>0.000000</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>590539</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>54.037533</td>
      <td>17.901295</td>
      <td>57.724444</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>279.95</td>
      <td>0.561057</td>
      <td>146.058108</td>
      <td>41.63895</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>590540 rows × 28 columns</p>
</div>




```python
DataCat
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
      <th>ProductCD</th>
      <th>card4</th>
      <th>card6</th>
      <th>P_emaildomain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>W</td>
      <td>discover</td>
      <td>credit</td>
      <td>gmail.com</td>
    </tr>
    <tr>
      <th>1</th>
      <td>W</td>
      <td>mastercard</td>
      <td>credit</td>
      <td>gmail.com</td>
    </tr>
    <tr>
      <th>2</th>
      <td>W</td>
      <td>visa</td>
      <td>debit</td>
      <td>outlook.com</td>
    </tr>
    <tr>
      <th>3</th>
      <td>W</td>
      <td>mastercard</td>
      <td>debit</td>
      <td>yahoo.com</td>
    </tr>
    <tr>
      <th>4</th>
      <td>H</td>
      <td>mastercard</td>
      <td>credit</td>
      <td>gmail.com</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>590535</th>
      <td>W</td>
      <td>visa</td>
      <td>debit</td>
      <td>gmail.com</td>
    </tr>
    <tr>
      <th>590536</th>
      <td>W</td>
      <td>mastercard</td>
      <td>debit</td>
      <td>gmail.com</td>
    </tr>
    <tr>
      <th>590537</th>
      <td>W</td>
      <td>mastercard</td>
      <td>debit</td>
      <td>gmail.com</td>
    </tr>
    <tr>
      <th>590538</th>
      <td>W</td>
      <td>mastercard</td>
      <td>debit</td>
      <td>aol.com</td>
    </tr>
    <tr>
      <th>590539</th>
      <td>W</td>
      <td>mastercard</td>
      <td>credit</td>
      <td>gmail.com</td>
    </tr>
  </tbody>
</table>
<p>590540 rows × 4 columns</p>
</div>




```python
y = DataRaw[ 'isFraud' ]
```

# Parte 1: Limpieza de los datos

En primer lugar cabe aclarar que los datos faltantes para cada variable se rellenaron con sus respectivas modas o medias según la naturaleza de la variable. En el caso de las categóricas, se omitieron aquellas variables cuyo porcentaje de datos faltantes superaba el 20%, ya que reemplazar estos datos faltantes por la moda podría llegar 

Para las variables continuas se usó el script provisto en el archivo `cleaning.py`, que consiste en dos fases de eliminación en base al mayor VIF en caso que este sea superior a 10 o 5. La primera fase ubica las dos variables independientes más correlacionadas entre sí y elimina aquella con mayor VIF en caso que supere el valor de 10. Este proceso para en el momento que la mayor correlación sea inferior a 0.5, despues de esto, se ejecuta la eliminación simple por VIF en el mayor de todos hasta que el mayor de los VIF sea inferior a 105. En últimas, esto garantiza una multicolinealidad extremadamente baja. 

Por otro lado, la eliminación en las variables categóricas se realizó mediante una prueba de contingencia de $\chi^2$, específicamente, se usó una matriz para visualizar el p-valor arrojado por la prueba entre las variables. Se pueden ver los resultados mencionados en la siguiente tabla:


```python
chisq_matrix( concat( [ DataFrame( y, columns=['isFraud'] ), DataCat ], axis=1 ) )
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
      <th>isFraud</th>
      <th>ProductCD</th>
      <th>card4</th>
      <th>card6</th>
      <th>P_emaildomain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>isFraud</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>7.129275e-79</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ProductCD</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>card4</th>
      <td>7.129275e-79</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>card6</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>P_emaildomain</th>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Es evidente que todas estas variables están fuertemente correlacionadas tanto entre sí mismas, como con la variable respuesta, por lo que es sufiente tomar una de estas para los modelos. En este caso se tomará la variable `ProductCD`, ya que no presentó datos faltantes (Esto se probará más adelante).

### Procentaje de datos faltantes

A continuación se pueden ver el porcentaje de los datos faltantes para las variables seleccionadas.


```python
from pandas import merge, read_csv, DataFrame
DataMissing = merge(read_csv('train_transaction.csv'), read_csv('train_identity.csv'), on='TransactionID', how='left')[[ 
    'isFraud', 'C1', 'C3', 'C5', 'D1', 'D3', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 
    'ProductCD', "V142", "V141", "V138", "V135", "V131", "V130", "V12", "V104",
    "V101", "V100", "V10", "TransactionAmt", "D9", "D8", "D7", "D5", "D4" ]]

Miss = DataFrame((DataMissing.isnull().sum() / DataMissing.shape[0])*100, columns=['%'])
Miss[ Miss['%'] >= 20 ].T.style.set_caption('Porcentaje de datos faltantes mayor o igual al 20%')
```




<style type="text/css">
</style>
<table id="T_52484">
  <caption>Porcentaje de datos faltantes mayor o igual al 20%</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_52484_level0_col0" class="col_heading level0 col0" >D3</th>
      <th id="T_52484_level0_col1" class="col_heading level0 col1" >D11</th>
      <th id="T_52484_level0_col2" class="col_heading level0 col2" >D12</th>
      <th id="T_52484_level0_col3" class="col_heading level0 col3" >D13</th>
      <th id="T_52484_level0_col4" class="col_heading level0 col4" >D14</th>
      <th id="T_52484_level0_col5" class="col_heading level0 col5" >V142</th>
      <th id="T_52484_level0_col6" class="col_heading level0 col6" >V141</th>
      <th id="T_52484_level0_col7" class="col_heading level0 col7" >V138</th>
      <th id="T_52484_level0_col8" class="col_heading level0 col8" >V10</th>
      <th id="T_52484_level0_col9" class="col_heading level0 col9" >D9</th>
      <th id="T_52484_level0_col10" class="col_heading level0 col10" >D8</th>
      <th id="T_52484_level0_col11" class="col_heading level0 col11" >D7</th>
      <th id="T_52484_level0_col12" class="col_heading level0 col12" >D5</th>
      <th id="T_52484_level0_col13" class="col_heading level0 col13" >D4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_52484_level0_row0" class="row_heading level0 row0" >%</th>
      <td id="T_52484_row0_col0" class="data row0 col0" >44.514851</td>
      <td id="T_52484_row0_col1" class="data row0 col1" >47.293494</td>
      <td id="T_52484_row0_col2" class="data row0 col2" >89.041047</td>
      <td id="T_52484_row0_col3" class="data row0 col3" >89.509263</td>
      <td id="T_52484_row0_col4" class="data row0 col4" >89.469469</td>
      <td id="T_52484_row0_col5" class="data row0 col5" >86.123717</td>
      <td id="T_52484_row0_col6" class="data row0 col6" >86.123717</td>
      <td id="T_52484_row0_col7" class="data row0 col7" >86.123717</td>
      <td id="T_52484_row0_col8" class="data row0 col8" >47.293494</td>
      <td id="T_52484_row0_col9" class="data row0 col9" >87.312290</td>
      <td id="T_52484_row0_col10" class="data row0 col10" >87.312290</td>
      <td id="T_52484_row0_col11" class="data row0 col11" >93.409930</td>
      <td id="T_52484_row0_col12" class="data row0 col12" >52.467403</td>
      <td id="T_52484_row0_col13" class="data row0 col13" >28.604667</td>
    </tr>
  </tbody>
</table>





```python
Miss[ Miss['%'] < 20 ].T.style.set_caption('Porcentaje de datos faltantes menor al 20%')
```




<style type="text/css">
</style>
<table id="T_5788e">
  <caption>Porcentaje de datos faltantes menor al 20%</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_5788e_level0_col0" class="col_heading level0 col0" >isFraud</th>
      <th id="T_5788e_level0_col1" class="col_heading level0 col1" >C1</th>
      <th id="T_5788e_level0_col2" class="col_heading level0 col2" >C3</th>
      <th id="T_5788e_level0_col3" class="col_heading level0 col3" >C5</th>
      <th id="T_5788e_level0_col4" class="col_heading level0 col4" >D1</th>
      <th id="T_5788e_level0_col5" class="col_heading level0 col5" >D10</th>
      <th id="T_5788e_level0_col6" class="col_heading level0 col6" >D15</th>
      <th id="T_5788e_level0_col7" class="col_heading level0 col7" >ProductCD</th>
      <th id="T_5788e_level0_col8" class="col_heading level0 col8" >V135</th>
      <th id="T_5788e_level0_col9" class="col_heading level0 col9" >V131</th>
      <th id="T_5788e_level0_col10" class="col_heading level0 col10" >V130</th>
      <th id="T_5788e_level0_col11" class="col_heading level0 col11" >V12</th>
      <th id="T_5788e_level0_col12" class="col_heading level0 col12" >V104</th>
      <th id="T_5788e_level0_col13" class="col_heading level0 col13" >V101</th>
      <th id="T_5788e_level0_col14" class="col_heading level0 col14" >V100</th>
      <th id="T_5788e_level0_col15" class="col_heading level0 col15" >TransactionAmt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5788e_level0_row0" class="row_heading level0 row0" >%</th>
      <td id="T_5788e_row0_col0" class="data row0 col0" >0.000000</td>
      <td id="T_5788e_row0_col1" class="data row0 col1" >0.000000</td>
      <td id="T_5788e_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_5788e_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_5788e_row0_col4" class="data row0 col4" >0.214888</td>
      <td id="T_5788e_row0_col5" class="data row0 col5" >12.873302</td>
      <td id="T_5788e_row0_col6" class="data row0 col6" >15.090087</td>
      <td id="T_5788e_row0_col7" class="data row0 col7" >0.000000</td>
      <td id="T_5788e_row0_col8" class="data row0 col8" >0.053172</td>
      <td id="T_5788e_row0_col9" class="data row0 col9" >0.053172</td>
      <td id="T_5788e_row0_col10" class="data row0 col10" >0.053172</td>
      <td id="T_5788e_row0_col11" class="data row0 col11" >12.881939</td>
      <td id="T_5788e_row0_col12" class="data row0 col12" >0.053172</td>
      <td id="T_5788e_row0_col13" class="data row0 col13" >0.053172</td>
      <td id="T_5788e_row0_col14" class="data row0 col14" >0.053172</td>
      <td id="T_5788e_row0_col15" class="data row0 col15" >0.000000</td>
    </tr>
  </tbody>
</table>




Como se puede ver en la tabla, gran parte de las variables elegidas al final, tienen un porcentaje de datos faltantes mayor al 20%. Para mantener el sentido del modelo, es mejor eliminar estas variables, ya que gran parte se reemplazará por la media, lo que puede llegar a generar datos irrelevantes para los modelos. Finalmente, los datos independientes quedan de la siguiente manera:


```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder( sparse_output = False ).set_output( transform = 'pandas' )
transformed = encoder.fit_transform( DatRawCat[[ 'ProductCD' ]] )
encoded = DataFrame( transformed, columns = encoder.get_feature_names_out(['ProductCD']) )
X = concat( [ encoded, DataClean], axis = 1 )
X.drop( columns = ['D3', 'D11', 'D12', 'D13', 'D14', 'V142', 'V141', 'V138', 'V10', 'D9', 'D8', 'D7', 'D5', 'D4'], inplace=True )
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
      <th>ProductCD_C</th>
      <th>ProductCD_H</th>
      <th>ProductCD_R</th>
      <th>ProductCD_S</th>
      <th>ProductCD_W</th>
      <th>C1</th>
      <th>C3</th>
      <th>C5</th>
      <th>D1</th>
      <th>D10</th>
      <th>D15</th>
      <th>V135</th>
      <th>V131</th>
      <th>V130</th>
      <th>V12</th>
      <th>V104</th>
      <th>V101</th>
      <th>V100</th>
      <th>TransactionAmt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>68.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>315.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>112.0</td>
      <td>84.000000</td>
      <td>111.000000</td>
      <td>0.0</td>
      <td>135.0</td>
      <td>354.0</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>123.982137</td>
      <td>163.744579</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.559711</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
with open( 'DataML.pkl', 'wb' ) as f:
    dump( y, f )
    dump( X, f )
```


```python
from pickle import load, dump

with open( 'DataML.pkl', 'rb' ) as f:
    y = load( f )
    X = load( f )
```

# Parte 2: Análisis exploratorio de los datos

### Variable dependiente


```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

y_df = pd.DataFrame(y, columns=['isFraud'])

value_counts = y_df['isFraud'].value_counts()

value_counts_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values})

plt.figure(figsize=(8, 6))
sns.barplot(x='Value', y='Count', data=value_counts_df, palette='Set2')
plt.title('Gráfico de barras de fraude')
plt.xlabel('Fraude')
plt.ylabel('Frecuencia')
plt.grid(True, axis='y')
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_18_0.png)
    


Como se puede ver en la gráfica, la variable `isFraud` cuenta con un total de 569877 (97%) de transacciones lícitas, por lo que puede ser problemático para la generación de modelos sin una forma de balancear la desproporción. 

### `ProductCD`


```python
value_counts = DataCat['ProductCD'].value_counts()

value_counts_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values})

plt.figure(figsize=(8, 6))
sns.barplot(x='Value', y='Count', data=value_counts_df, palette='Set2')
plt.title('Gráfico de barras de código de producto')
plt.xlabel('Fraude')
plt.ylabel('Frecuencia')
plt.grid(True, axis='y')
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_21_0.png)
    



```python
from pandas import crosstab
from numpy import zeros, array, append

eda_df = concat( [ y_df, DataCat ], axis = 1 )

cross_table = crosstab( eda_df['isFraud'], eda_df['ProductCD'] )

ct = dict()
cross_dict = cross_table.to_dict()
for key in cross_dict:
    item = array([])
    for sub_key in cross_dict[ key ]: item = append( item, cross_dict[ key ][ sub_key ] )
    ct[ key ] = item

isFraud = ( '0', '1' )

fig, ax = plt.subplots(figsize=(8, 6))
bottom = zeros(2)

for boolean, count in ct.items():
    p = ax.bar(isFraud, count, label=boolean, bottom=bottom )
    bottom += count
ax.set_title("Código de producto según fraude detectado")
ax.legend(loc="upper right")

plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_22_0.png)
    


## Variables contínuas

Se puede observar el resumen de las variables en la siguiente tabla


```python
X.drop( columns = [ 'ProductCD_C', 'ProductCD_H', 'ProductCD_R', 'ProductCD_S', 'ProductCD_W' ] ).describe()
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
      <th>C1</th>
      <th>C3</th>
      <th>C5</th>
      <th>D1</th>
      <th>D10</th>
      <th>D15</th>
      <th>V135</th>
      <th>V131</th>
      <th>V130</th>
      <th>V12</th>
      <th>V104</th>
      <th>V101</th>
      <th>V100</th>
      <th>TransactionAmt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
      <td>590540.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.092458</td>
      <td>0.005644</td>
      <td>5.571526</td>
      <td>94.347568</td>
      <td>123.982137</td>
      <td>163.744579</td>
      <td>17.250132</td>
      <td>31.133302</td>
      <td>92.165849</td>
      <td>0.559711</td>
      <td>0.085433</td>
      <td>0.889249</td>
      <td>0.273504</td>
      <td>135.027176</td>
    </tr>
    <tr>
      <th>std</th>
      <td>133.569018</td>
      <td>0.150536</td>
      <td>25.786976</td>
      <td>157.490898</td>
      <td>170.456102</td>
      <td>186.805646</td>
      <td>293.769431</td>
      <td>161.118406</td>
      <td>315.876473</td>
      <td>0.476516</td>
      <td>0.648545</td>
      <td>20.577098</td>
      <td>0.946924</td>
      <td>239.162522</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-83.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.251000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>43.321000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>43.000000</td>
      <td>117.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.559711</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>68.769000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>121.000000</td>
      <td>150.000000</td>
      <td>251.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>59.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>125.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4685.000000</td>
      <td>26.000000</td>
      <td>349.000000</td>
      <td>640.000000</td>
      <td>876.000000</td>
      <td>879.000000</td>
      <td>90750.000000</td>
      <td>55125.000000</td>
      <td>55125.000000</td>
      <td>3.000000</td>
      <td>15.000000</td>
      <td>869.000000</td>
      <td>28.000000</td>
      <td>31937.391000</td>
    </tr>
  </tbody>
</table>
</div>



Es evidente que para la mayoria de variables, la desviación estándar es mucho más alta que la media, por lo que los datos van a estar extremadamente dispersos. Más aún, Todos los máximos son extremadamente altos, mucho más que el tercer cuartil, por tanto se evidencian datos extremadamente atípicos. También es notable la diferencia entre la media y la mediana, lo que implica que todas las variables tienen una severa asimetría.

### `TransactionAmt`


```python
transaction_amt_values = X['TransactionAmt']

plt.figure(figsize=(10, 6))
plt.hist(transaction_amt_values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma del monto')
plt.xlabel('Monto de la transacción')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_27_0.png)
    


Es notable que casi el total de los datos se encuentra en el primer intérvalo inferior a 5000, además el tamaño del intérvalo implica la existencia de uno o más datos atípicos extremadamente altos.


```python
eda_df = concat( [eda_df, X], axis = 1 )
```


```python
eda_df['isFraud'] = eda_df['isFraud'].astype(str)

plt.figure(figsize=(10, 6))
sns.boxplot(x='TransactionAmt', y='isFraud', data=eda_df, palette='Set2' )
plt.title('Cantidad del monto según fraude detectado')
plt.xlabel('Monto')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_30_0.png)
    


Como se mencionó previamente, se encuentra un dato extremádamente atípico. Podemos ver que hay una tendencia para las transacciones lícitas en alcanzar valores más altos, aún así, no se pueden sacar muchas conclusiones dada la desproporción de los datos.

### `C1`


```python
values = X['C1']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de C1')
plt.xlabel('C1')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_33_0.png)
    


Así como en la variable anterior, se encuentra una asimetría extrema. Esto se debe especialmente al hecho que el 75% de los datos es menor o igual a 3, mientras que hay un rango extremadamente amplio.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='C1', y='isFraud', data=eda_df, palette='Set2' )
plt.title('C1 según fraude detectado')
plt.xlabel('C1')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_35_0.png)
    


En este caso tambien es evidente que hay mayor concentración de datos atípicos en el conjunto lícito dada la disparidad, pero se puede ver claramente que ambas mantienen cierta similitud. Como se pudo ver en la tabla descriptiva, la gran mayoria de las variables se concentran en el intérvalo de 0-3.

### `C3`


```python
values = X['C3']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de C3')
plt.xlabel('C3')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_38_0.png)
    


Es notorio que no se puede apreciar ningún valor por fuera del intérvalo marcado, por lo que como el resto de variables, es evidencia de muchos datos atípicos.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='C3', y='isFraud', data=eda_df, palette='Set2' )
plt.title('C3 según fraude detectado')
plt.xlabel('C3')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_40_0.png)
    


Como se menciona antes, la mayoria de datos se concentra en el valor de 0, aún así, se presentan unos cuantos datos atípicos en la parte lícita de las transacciones.

### `C5`


```python
values = X['C5']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de C5')
plt.xlabel('C5')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_43_0.png)
    


A diferencia de las variables anteriores, esta sí permite observar unos cuantos valores por fuera de los valores iniciales, aún así, mantiene el patrón previo de una concentración de datos extrema al inicio del intérvalo.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='C5', y='isFraud', data=eda_df, palette='Set2' )
plt.title('C5 según fraude detectado')
plt.xlabel('C5')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_45_0.png)
    


Por otro lado, podemos observar una ligera diferencia más marcada entre los casos de fraude, ya que las transacciones lícitas tienen en general datos atípicos mayores que los de las transacciones marcadas como fraude.

### `D1`


```python
values = X['D1']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de D1')
plt.xlabel('D1')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_48_0.png)
    


De forma similar a la variable anterior, a pesar de tener datos atípicos, es evidente que la variable posee una distribución más cercana al origen.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='D1', y='isFraud', data=eda_df, palette='Set2' )
plt.title('D1 según fraude detectado')
plt.xlabel('D1')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_50_0.png)
    


Es evidente que las transacciones lícitas tienen tendencia a poseer un valor mayor de la variable D1.

### `D10`


```python
values = X['D10']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('D10')
plt.xlabel('D10')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_53_0.png)
    


Esta variable presenta una distribución aún menos dispersa que la anterior, por lo que se puede esperar una mejor representación de la variable respuesta.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='D10', y='isFraud', data=eda_df, palette='Set2' )
plt.title('D10 según fraude detectado')
plt.xlabel('D10')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_55_0.png)
    


Es evidente que la mediana de las transacciones lícitas es mayor a aquella de las ilícitas, mientras que los datos atípicos se extienden a un mayor rango.

### `D15`


```python
values = X['D15']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de D15')
plt.xlabel('D15')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_58_0.png)
    


Esta variable comparte comportamiento con la variable anterior, presentando aún menos dispersión de los datos.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='D15', y='isFraud', data=eda_df, palette='Set2' )
plt.title('D15 según fraude detectado')
plt.xlabel('D15')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_60_0.png)
    


De la misma forma, hay una diferencia marcada en la mediana para ambas categorías.

### `V100`


```python
values = X['V100']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de V100')
plt.xlabel('V100')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_63_0.png)
    


Por otro lado, esta variable presenta un comportamiento similar a las primeras variables, donde se espera una gran cantidad de datos atípicos.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='V100', y='isFraud', data=eda_df, palette='Set2' )
plt.title('V100 según fraude detectado')
plt.xlabel('V100')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_65_0.png)
    


Aún así, esta variable presenta una ligera diferencia para la variable categórica, dado que los datos atípicos ( o mayores a 0 ), se extienden más para las transacciones lícitas

### `V131`


```python
values = X['V131']

plt.figure(figsize=(10, 6))
plt.hist(values, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de V131')
plt.xlabel('V131')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_68_0.png)
    


Esta variable presenta el mismo comportamiento a la primera variable, donde no son evidentes valores por fuera de un intervalo, lo que implica que hay valores atípicos extremadamente grandes.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='V131', y='isFraud', data=eda_df, palette='Set2' )
plt.title('V131 según fraude detectado')
plt.xlabel('V131')
plt.ylabel('Fraude')
plt.grid(True)
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_70_0.png)
    


Como se menciona antes, la mayoría de datos se concentran en el valor 0, teniendo unos cuantos datos atípicos más grandes en las transacciones lícitas.

# Parte 3: Modelos de clasificación

En primer lugar, para evitar fuga de información, se separará un conjunto de prueba, dado el uso de ADASYN


```python
from sklearn.model_selection import train_test_split

y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 37 )

y_df = pd.DataFrame(y_test, columns=['isFraud'])

value_counts = y_df['isFraud'].value_counts()

value_counts_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values})

plt.figure(figsize=(8, 6))
sns.barplot(x='Value', y='Count', data=value_counts_df, palette='Set2')
plt.title('Gráfico de barras de fraude muestreado')
plt.xlabel('Fraude')
plt.ylabel('Frecuencia')
plt.grid(True, axis='y')
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_74_0.png)
    


Como se puede ver, el conjunto de testeo mantiene la proporción antes mencionada de 97-3. Ahora bien, se realizarán los modelos con y sin rebalanceamiento ADASYN. Podemos visualizar la nueva variable dependiente balanceada con la siguiente gráfica:


```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN( random_state = 37 )

X_bal, y_bal = adasyn.fit_resample( X_train, y_train )
```


```python
y_df = pd.DataFrame(y_bal, columns=['isFraud'])

value_counts = y_df['isFraud'].value_counts()

value_counts_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values})

plt.figure(figsize=(8, 6))
sns.barplot(x='Value', y='Count', data=value_counts_df, palette='Set2')
plt.title('Gráfico de barras de fraude rebalanceado')
plt.xlabel('Fraude')
plt.ylabel('Frecuencia')
plt.grid(True, axis='y')
plt.show()
```


    
![png](Ejercicio3_files/Ejercicio3_77_0.png)
    



```python
with open( 'DataModels.pkl', 'wb' ) as f:
    dump( ( X_train, X_bal, X_test ), f )
    dump( ( y_train, y_bal, y_test ), f )
```

### Clasificación Bayesiana

### Árboles de decisión

### Random Forest

### XGBoost

### K-NN

### Regresión logística con penalización

## Comparación de modelos


```python

```
