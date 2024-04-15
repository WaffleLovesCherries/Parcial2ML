# Resumen

ADASYN o *Adaptive Synthetic Sampling Approach* es un método de balanceamiento para un conjunto de datos donde las categorías objetivo presentan una gran desproporción en comparación a otras, evitando problemas como sesgos hacia la variable mayoritaria por parte de los modelos generados. Esto lo consigue mediante un remuestreo de los datos similar a SMOTE, generando nuevos datos entre dos pertenecientes a la clase minoritaria, con la excepción de que tiene preferencia por ubicar nuevos datos en zonas donde se presenta una mayor cantidad de datos de otras clases.

# Metodología

El algorítmo ADASYN consiste en asignar un peso a cada dato minoritario, cuyo valor depende del número de vecinos de categorías diferentes a la minoritaria. La fórmula es la siguiente

$$ w_i = \dfrac{ \# \text{diff}_i }{k} $$

Siendo $k$ el número de vecinos con el que se va a trabajar. Nótese que todos aquellos nodos cuyos pesos sean $0$, implicarán que todos los vecinos son de su misma clase. Dado que zonas donde se agrupan una gran cantidad de datos de la misma categoría facilitan la detección de patrones para el modelo, no es necesario introducir nuevos puntos. Por otro lado, aquellas zonas donde se encuentren diferentes categorías serán en las que el modelo necesite más información. 

Por tanto, para todo punto con $w_i > 0$, se ubican puntos entre este y el punto de la misma categoría más cercano. de forma similar al método SMOTE.

# Librería python

Parámetros de `imblearn.over_sampling.ADASYN`:

- `sampling_strategy`: 

    - Cuando es de tipo `float`, indica la proporción buscada entre la clase minoritaria y la mayoritaria.
    - Cuando es de tipo `str`, identifica la clase objetivo para remuestrear.
    - Si se quiere usar una combinación de estos se puede usar un diccionario.

- `random_state`: La semilla aleatorea para el proceso.

- `n_neighbors`: El $k$ antes mencionado, es decir, el número de vecinos que cada nodo minoritario va a tener. El valor por defecto es 5.

- `n_jobs`: El número de núcleos que se usarán para el proceso.

Métodos:

- `fit_resample`: El único método que realmente importa, remuestrea el conjunto de datos con el proceso antes dicho y retorna la tupla correspondiente.

- `fit`: Revisa los inputs y estadísticas del muestreador.

- `get_feature_names_out`: Obtiene las características importantes del muestreo.

