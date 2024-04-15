# Resumen

La optimización bayesiana es un método de optimización para funciones cuyo funcionamiento interno es desconocido, también conocidos como caja negra. En particular, la mayoría de hiperparametrizaciones en Machile Learning cuentan con una o mas variables a optimizar, pero que desconocemos su afectación en la métrica hasta que sea probada.

# Metodología

El algoritmo de optimización bayesiana consiste en muestrear un conjunto de datos iniciales de la función objetivo, de los cuales se predice una función. Esta función irá cambiando a medida que se agreguen más puntos, y se detendrá en el momento que se obvserve que la función predicha no cambie al añadirse nuevos puntos, o la varianza de la función caiga por debajo de un límite establecido. Todo esto parte del supuesto que los puntos de la función se distribuyen de forma normal e independiente del tiempo. Esta función predicha se obtiene interpolando los puntos con una varianza fija, es decir, se obtienen varias funciones interpoladoras, de las cuales se asume una distribución normal. Finalmente la función predicha se obtiene promediando todos estos polinomios.

# Librería python

Parámetros de `bayes_opt.BayesianOptimization`:

- `f`: La función cualquiera a optimizar, debe retornar un número real.
- `pbounds`: Un diccionario de tuplas donde cada parámetro de la variable está acompañado del intervalo en el que se permiten valores.
- `random_state`: Semilla aleatorea.

Una vez inicializado el optimizador, se maximiza la función por medio del método `maximize` con los siguientes parámetros:
- `init_points`: Puntos iniciales que se usarán en la función.
- `n_iter`: Numero de pasos de exploración aleatoria en el proceso de maximización, un mayor número implica un mayor rango de búsqueda.

