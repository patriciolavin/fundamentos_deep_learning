# Proyecto 3: Predicción de Natalidad con Redes Neuronales

**Tags:** `Deep Learning`, `Redes Neuronales`, `Clasificación`, `Keras`, `TensorFlow`, `Python`

## Objetivo del Proyecto

Este proyecto explora la aplicación del **Deep Learning** para resolver un problema de clasificación complejo: predecir la tasa de natalidad (categorizada como alta, media o baja) basándose en un conjunto de factores socioeconómicos. El objetivo es construir y entrenar una **Red Neuronal Densa (DNN)** capaz de aprender patrones no lineales en los datos para lograr una clasificación precisa.

## Metodología y Herramientas

La solución se construyó utilizando el ecosistema de Deep Learning de Python:

1.  **Preprocesamiento Avanzado:** Además de la limpieza de datos con `Pandas`, se realizó una codificación **One-Hot** de las variables categóricas para prepararlas para la entrada a la red neuronal. Los datos numéricos fueron estandarizados.
2.  **Arquitectura del Modelo:**
    * Se diseñó una Red Neuronal secuencial utilizando la API de **Keras** con `TensorFlow` como backend.
    * La arquitectura consiste en una capa de entrada, **[Número] capas ocultas** con función de activación **ReLU** (Rectified Linear Unit) para introducir no linealidad, y una capa de salida con activación **Softmax**, ideal para clasificación multiclase.
    * Se incluyeron capas de **Dropout** para prevenir el sobreajuste.
3.  **Compilación y Entrenamiento:**
    * El modelo se compiló con el optimizador **Adam** y la función de pérdida `categorical_crossentropy`.
    * Se entrenó durante **[Número] épocas**, monitoreando tanto la precisión (accuracy) como la pérdida (loss) en los conjuntos de entrenamiento y validación.
4.  **Evaluación:** El rendimiento final se evaluó en un conjunto de prueba no visto, utilizando un **reporte de clasificación** y una **matriz de confusión** para analizar la precisión por clase.

## Resultados Clave

La red neuronal entrenada alcanzó una **precisión de clasificación del [Tu %]** en el conjunto de prueba. La matriz de confusión demostró que el modelo es particularmente efectivo en identificar la clase [Nombre de la clase mejor predicha], validando el poder de las redes neuronales para capturar relaciones complejas que otros modelos podrían pasar por alto.

## Cómo Utilizar

1.  Clona este repositorio: `git clone https://github.com/patriciolavin/fundamentos_deep_learning.git`
2.  Instala las dependencias: `pip install pandas scikit-learn tensorflow`
3.  Ejecuta la Jupyter Notebook para seguir el proceso de construcción y entrenamiento del modelo.
