# Proyecto 3: Predicción de Natalidad con Redes Neuronales

**Tags:** `Deep Learning`, `Redes Neuronales`, `Clasificación`, `Keras`, `TensorFlow`, `Python`

## Objetivo del Proyecto

Este proyecto explora la aplicación del **Deep Learning** para resolver un problema de clasificación complejo: predecir la tasa de natalidad (categorizada como alta, media o baja) basándose en un conjunto de factores socioeconómicos. El objetivo es construir y entrenar una **Red Neuronal Densa (DNN)** capaz de aprender patrones no lineales en los datos para lograr una clasificación precisa.

## Metodología y Herramientas

La solución se construyó utilizando el ecosistema de Deep Learning de Python:

1.  **Preprocesamiento Avanzado:** Además de la limpieza de datos con `Pandas`, se realizó una codificación **One-Hot** de las variables categóricas para prepararlas para la entrada a la red neuronal. Los datos numéricos fueron estandarizados.
2.  **Arquitectura del Modelo:**
    * Se diseñó una Red Neuronal secuencial utilizando la API de **Keras** con `TensorFlow` como backend.
    * La arquitectura consiste en una capa de entrada, varias capas ocultas con función de activación **ReLU** y una capa de salida con activación **Softmax**, ideal para clasificación multiclase.
    * Se incluyeron capas de **Dropout** para prevenir el sobreajuste.
3.  **Compilación y Entrenamiento:**
    * El modelo se compiló con el optimizador **Adam** y la función de pérdida `categorical_crossentropy`.
    * Se entrenó durante varias épocas, monitoreando tanto la precisión (accuracy) como la pérdida (loss) en los conjuntos de entrenamiento y validación.
4.  **Evaluación:** El rendimiento final se evaluó en un conjunto de prueba no visto, utilizando un **reporte de clasificación** y una **matriz de confusión** para analizar la precisión por clase.

## Resultados Clave

La red neuronal entrenada alcanzó una alta precisión en la clasificación en el conjunto de prueba. La matriz de confusión demostró que el modelo es particularmente efectivo en identificar las diferentes clases, validando el poder de las redes neuronales para capturar relaciones complejas que otros modelos podrían pasar por alto.

## Reflexión Personal y Desafíos

Este fue mi primer proyecto formal con Deep Learning y se sintió como pasar a un nuevo nivel de abstracción y poder. El requisito era claro: usar un modelo más allá de Scikit-learn para un problema de clasificación.

* **Punto Alto:** Diseñar la arquitectura de la red en Keras. Apilar `Dense`, `Dropout` y `Activation` se siente como construir con bloques de LEGO increíblemente potentes. Ver las curvas de `loss` y `accuracy` moverse en la dirección correcta durante el entrenamiento es adictivo; es una retroalimentación en tiempo real de que tu creación está "aprendiendo".
* **Punto Bajo:** El miedo a la "caja negra". Con una red neuronal, es más difícil responder al "porqué" de una predicción en comparación con un modelo más simple. Además, el primer encuentro con el sobreajuste (overfitting) fue una lección de humildad. Ver cómo la precisión en el set de validación se estancaba o empeoraba mientras la del entrenamiento seguía subiendo fue un llamado de atención sobre la importancia de la regularización (como Dropout) y de no confiar ciegamente en las métricas de entrenamiento.

## Cómo Utilizar

1.  Clona este repositorio: `git clone https://github.com/patriciolavin/fundamentos_deep_learning.git`
2.  Instala las dependencias: `pip install pandas scikit-learn tensorflow`
3.  Ejecuta la Jupyter Notebook para seguir el proceso de construcción y entrenamiento del modelo.
