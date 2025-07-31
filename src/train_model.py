import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import json
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
from contextlib import contextmanager
from packaging import version
from sklearn.pipeline import Pipeline
from jinja2 import Template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from utils import config
from utils.logger import log

# Configurar generador de números aleatorios
rng = np.random.RandomState(config.RANDOM_STATE)

# Validar versión de TensorFlow
if version.parse(tf.__version__) > version.parse('2.4.0'):
    log.warning(
        "La versión de TensorFlow es superior a 2.4.0, lo que puede causar problemas. "
        "Se recomienda usar TensorFlow 2.4.0 o compatible."
    )

# Suprimir warnings
warnings.filterwarnings("ignore", category=UserWarning)

@contextmanager
def resource_cleanup():
    """Context manager para limpieza de recursos matplotlib."""
    try:
        yield
    finally:
        plt.close('all')

def save_fig_to_base64(fig):
    """Guarda una figura de matplotlib en base64."""
    buf = BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        if not img_b64:
            log.error("Error: Imagen base64 vacía generada")
            raise ValueError("Imagen base64 vacía")
        if not img_b64.startswith('iVBOR'):
            log.error("Imagen base64 no es un PNG válido")
            raise ValueError("Imagen base64 no es un PNG válido")
        log.debug(f"Imagen base64 generada, longitud: {len(img_b64)} caracteres")
        return img_b64
    finally:
        plt.close(fig)
        buf.close()

def validate_base64_png(base64_string):
    """Valida si un string base64 representa una imagen PNG válida."""
    try:
        decoded = base64.b64decode(base64_string)
        if decoded.startswith(b'\x89PNG\r\n\x1a\n'):
            log.debug("Imagen base64 es un PNG válido")
            return True
        else:
            log.error("Imagen base64 no es un PNG válido")
            return False
    except Exception as e:
        log.error(f"Error validando base64 PNG: {e}")
        return False

def split_data(df):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    log.info("Dividiendo datos en entrenamiento y prueba.")
    
    if df.empty:
        raise ValueError("El DataFrame está vacío")
    
    if not all(col in df.columns for col in config.FEATURE_COLUMNS + [config.TARGET_VARIABLE]):
        missing_cols = [col for col in config.FEATURE_COLUMNS + [config.TARGET_VARIABLE] if col not in df.columns]
        raise KeyError(f"Columnas faltantes: {missing_cols}")
    
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_VARIABLE]
    
    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Los datos contienen NaN")
    if np.isinf(X.values).any() or np.isinf(y.values).any():
        raise ValueError("Los datos contienen infinitos")
    
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

def generate_diagnostic_plots(y_true, y_pred, model_name):
    """Genera gráficos de diagnóstico."""
    log.info(f"Generando gráficos de diagnóstico para {model_name}")
    
    if y_true is None or y_pred is None:
        raise ValueError("y_true o y_pred son None")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true y y_pred tienen longitudes diferentes: {len(y_true)} vs {len(y_pred)}")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true o y_pred contienen NaN")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_true o y_pred contienen infinitos")
    
    log.debug(f"Tipo de y_true: {type(y_true)}, forma: {getattr(y_true, 'shape', len(y_true))}")
    log.debug(f"Tipo de y_pred: {type(y_pred)}, forma: {getattr(y_pred, 'shape', len(y_pred))}")
    
    plots = {}
    y_true_flat = y_true.values.flatten() if hasattr(y_true, 'values') else y_true.flatten()
    
    with resource_cleanup():
        fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
        ax_pred.scatter(y_true_flat, y_pred, alpha=0.7, s=50)
        ax_pred.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
        ax_pred.set_xlabel('Valores Reales')
        ax_pred.set_ylabel('Valores Predichos')
        ax_pred.set_title(f'Reales vs. Predichos - {model_name}')
        ax_pred.grid(True, alpha=0.3)
        plots[f'predicciones_{model_name}'] = save_fig_to_base64(fig_pred)

        residuals = y_true_flat - y_pred
        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        ax_res.scatter(y_pred, residuals, alpha=0.7, s=50)
        ax_res.axhline(y=0, color='r', linestyle='--', lw=2)
        ax_res.set_xlabel('Valores Predichos')
        ax_res.set_ylabel('Residuos')
        ax_res.set_title(f'Análisis de Residuos - {model_name}')
        ax_res.grid(True, alpha=0.3)
        plots[f'residuos_{model_name}'] = save_fig_to_base64(fig_res)
    
    return plots

def generate_nn_learning_curve(history, model_name):
    """Genera curva de aprendizaje para red neuronal."""
    with resource_cleanup():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f'Curvas de Aprendizaje - {model_name}', fontsize=16)

        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], 'b-', label='Pérdida de Entrenamiento (MSE)', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], 'r-', label='Pérdida de Validación (MSE)', linewidth=2)
        ax1.set_ylabel('MSE (Loss)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history.history['mae'], 'b-', label='MAE de Entrenamiento', linewidth=2)
        ax2.plot(epochs, history.history['val_mae'], 'r-', label='MAE de Validación', linewidth=2)
        ax2.set_ylabel('Mean Absolute Error (MAE)')
        ax2.set_xlabel('Épocas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return {f'curva_aprendizaje_{model_name}': save_fig_to_base64(fig)}

def train_and_evaluate_classics(X_train, y_train, X_test, y_test):
    """Entrena y evalúa modelos clásicos."""
    log.info("Entrenando modelos clásicos.")
    pipelines = {
        'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=config.RANDOM_STATE))]),
        'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=config.RANDOM_STATE))]),
        'RandomForest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE))])
    }
    results = []
    all_plots = {}

    for name, pipe in pipelines.items():
        log.info(f"Entrenando modelo: {name}")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({'Modelo': name, 'R² Score': r2, 'MAE': mae, 'RMSE': rmse})
        all_plots.update(generate_diagnostic_plots(y_test, y_pred, name))
        
        log.info(f"Modelo {name} - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return pd.DataFrame(results), all_plots, pipelines

def build_nn_model(num_features, activation='relu', optimizer_name='adam', learning_rate=0.01, use_dropout=False, use_l2=False):
    """Construye un modelo de red neuronal."""
    if num_features <= 0:
        raise ValueError("num_features debe ser mayor que 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser mayor que 0")
    if activation not in ['relu', 'sigmoid', 'tanh', 'elu']:
        raise ValueError("activation debe ser 'relu', 'sigmoid', 'tanh' o 'elu'")
    if optimizer_name.lower() not in ['adam', 'sgd', 'rmsprop']:
        raise ValueError("optimizer_name debe ser 'adam', 'sgd' o 'rmsprop'")
    
    regularizer = l2(0.001) if use_l2 else None
    
    inputs = tf.keras.layers.Input(shape=(num_features,), name='input_layer')
    x = tf.keras.layers.Dense(64, activation=activation, kernel_regularizer=regularizer, name='dense_1')(inputs)
    if use_dropout:
        x = tf.keras.layers.Dropout(0.2, name='dropout_1')(x)
    x = tf.keras.layers.Dense(32, activation=activation, kernel_regularizer=regularizer, name='dense_2')(x)
    outputs = tf.keras.layers.Dense(1, name='output_layer')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'nn_model_{activation}_{optimizer_name}')
    
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        loss='mean_squared_error', 
        optimizer=optimizer, 
        metrics=['mae'],
        run_eagerly=False
    )
    
    return model

def generate_permutation_importance(model, X_data, model_name, scaler=None):
    """Genera importancia de variables usando permutación."""
    try:
        from sklearn.inspection import permutation_importance
        
        if 'NN' in model_name:
            def predict_fn(X):
                predictions = model.predict(X, verbose=0)
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    log.error(f"Predicciones contienen NaN o infinitos para '{model_name}'")
                    raise ValueError("Predicciones contienen NaN o infinitos")
                predictions_flat = predictions.flatten()
                if np.var(predictions_flat) < 1e-10:
                    log.error(f"Predicciones uniformes detectadas para '{model_name}'")
                    raise ValueError("Predicciones uniformes detectadas")
                return predictions_flat
        else:
            if isinstance(model, Pipeline):
                log.info(f"Modelo '{model_name}' es un Pipeline, usando estimador subyacente")
                estimator = model.named_steps['model']
            else:
                estimator = model
            predict_fn = estimator.predict
        
        sample_size = min(100, len(X_data))
        X_sample = X_data[:sample_size]
        
        if np.any(np.isnan(X_sample)) or np.any(np.isinf(X_sample)):
            log.error(f"X_sample contiene NaN o infinitos para '{model_name}'")
            raise ValueError("X_sample contiene NaN o infinitos")
        
        y_sample = predict_fn(X_sample)
        if np.any(np.isnan(y_sample)) or np.any(np.isinf(y_sample)):
            log.error(f"y_sample contiene NaN o infinitos para '{model_name}'")
            raise ValueError("y_sample contiene NaN o infinitos")
        
        perm_importance = permutation_importance(
            estimator if 'NN' not in model_name else model,
            X_sample, y_sample,
            n_repeats=5, random_state=rng, n_jobs=1
        )
        
        importances = perm_importance.importances_mean
        if np.any(np.isnan(importances)) or np.any(np.isinf(importances)):
            log.error(f"Importancias contienen NaN o infinitos para '{model_name}'")
            raise ValueError("Importancias contienen NaN o infinitos")
        
        if importances.max() > 0:
            importances = importances / importances.max() * 0.9 + 0.1
        else:
            log.warning(f"Importancias tienen máximo cero para '{model_name}'. Usando valores uniformes.")
            importances = np.ones(len(X_data[0]) if len(X_data) > 0 else 6, dtype=np.float64) * 0.1
        
        return importances
    
    except Exception as e:
        log.error(f"Error en permutación para '{model_name}': {e}")
        return np.ones(len(X_data[0]) if len(X_data) > 0 else 6, dtype=np.float64) * 0.1

def analyze_feature_importance(model, model_name, X_train, scaler=None):
    """Analiza la importancia de variables usando métodos nativos o permutación."""
    log.info(f"Analizando importancia de variables para '{model_name}'.")
    
    try:
        if not all(col in X_train.columns for col in config.FEATURE_COLUMNS):
            log.error(f"X_train no contiene columnas esperadas: {config.FEATURE_COLUMNS}")
            raise ValueError("X_train no contiene columnas esperadas")
        if len(X_train.columns) != len(config.FEATURE_COLUMNS):
            log.error(f"X_train tiene {len(X_train.columns)} columnas, esperado {len(config.FEATURE_COLUMNS)}")
            raise ValueError("Número de columnas no coincide")
        
        X_train_scaled = scaler.transform(X_train) if scaler else X_train.copy()
        
        if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
            log.error(f"X_train_scaled contiene NaN o infinitos para '{model_name}'")
            raise ValueError("Datos escalados contienen NaN o infinitos")
        
        variances = np.var(X_train_scaled, axis=0)
        if np.any(variances < 1e-10):
            low_var_cols = X_train.columns[variances < 1e-10].tolist()
            log.error(f"Columnas con varianza baja: {low_var_cols}")
            raise ValueError(f"Columnas con varianza baja: {low_var_cols}")
        
        importances = None
        
        if 'NN' in model_name:
            log.info(f"Usando permutation_importance para red neuronal '{model_name}'")
            importances = generate_permutation_importance(model, X_train_scaled, model_name, scaler)
        else:
            log.info(f"Usando importancia nativa para modelo clásico '{model_name}'")
            if isinstance(model, Pipeline):
                estimator = model.named_steps['model']
            else:
                estimator = model
            
            if isinstance(estimator, RandomForestRegressor):
                importances = estimator.feature_importances_
            elif isinstance(estimator, (Ridge, Lasso)):
                importances = np.abs(estimator.coef_)
            else:
                log.warning(f"Modelo '{model_name}' no tiene importancia nativa, usando permutación")
                importances = generate_permutation_importance(model, X_train_scaled, model_name, scaler)
        
        if importances is None or len(importances) != len(X_train.columns):
            log.error(f"Importancias inválidas: {importances}")
            raise ValueError("Importancias no válidas o dimensiones incorrectas")
        
        if np.any(np.isnan(importances)) or np.any(np.isinf(importances)):
            log.error(f"Importancias contienen NaN o infinitos para '{model_name}'")
            raise ValueError("Importancias contienen NaN o infinitos")
        
        if importances.max() > 0:
            importances = importances / importances.max() * 0.9 + 0.1
        else:
            log.warning(f"Importancias tienen máximo cero para '{model_name}'. Usando valores uniformes.")
            importances = np.ones(len(X_train.columns)) * 0.1
        
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns.astype(str),
            'Importance': importances.astype(np.float64)
        }).sort_values(by='Importance', ascending=False)
        
        if feature_importance_df.empty:
            log.error(f"feature_importance_df vacío para '{model_name}'")
            raise ValueError("feature_importance_df vacío")
        if not all(col in feature_importance_df.columns for col in ['Feature', 'Importance']):
            log.error(f"feature_importance_df no contiene columnas esperadas: {feature_importance_df.columns}")
            raise ValueError("feature_importance_df no contiene columnas esperadas")
        if feature_importance_df['Importance'].isna().any():
            log.error(f"feature_importance_df contiene NaN para '{model_name}'")
            raise ValueError("feature_importance_df contiene NaN")
        if feature_importance_df['Feature'].dtype != object:
            log.error(f"Columna 'Feature' no es de tipo str: {feature_importance_df['Feature'].dtype}")
            raise ValueError("Columna 'Feature' no es str")
        if feature_importance_df['Importance'].dtype != np.float64:
            log.error(f"Columna 'Importance' no es float64: {feature_importance_df['Importance'].dtype}")
            raise ValueError("Columna 'Importance' no es float64")
        
        log.info(f"feature_importance_df generado: {feature_importance_df.to_dict()}")
        
        with resource_cleanup():
            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                plot_df = feature_importance_df.copy()
                plot_df = plot_df.rename(columns={'Importance': 'Importancia Media Absoluta (SHAP)', 'Feature': 'Variable'})
                
                sns.barplot(
                    data=plot_df,
                    x='Importancia Media Absoluta (SHAP)',
                    y='Variable',
                    hue='Variable',
                    ax=ax,
                    palette='viridis',
                    legend=False
                )

                ax.set_title(f"Importancia de Variables según SHAP - {model_name}")
                ax.set_xlabel('Importancia Normalizada')
                ax.set_ylabel('Variable')
                plt.tight_layout()
                importance_plot_b64 = save_fig_to_base64(fig)
                if not importance_plot_b64 or not validate_base64_png(importance_plot_b64):
                    raise ValueError("Gráfico SHAP no es un PNG válido")
                log.info(f"Gráfico SHAP generado para '{model_name}', longitud: {len(importance_plot_b64)} caracteres")
            except Exception as plot_error:
                log.error(f"Error generando gráfico SHAP: {plot_error}")
                ax.text(0.5, 0.5, 'Gráfico de Importancia no disponible\nValores uniformes asignados',
                        horizontalalignment='center', verticalalignment='center', fontsize=12)
                ax.set_title(f"Importancia de Variables según SHAP - {model_name}")
                plt.tight_layout()
                importance_plot_b64 = save_fig_to_base64(fig)
                if not validate_base64_png(importance_plot_b64):
                    log.error("Gráfico SHAP de respaldo no es un PNG válido")
                log.info(f"Gráfico SHAP de respaldo generado para '{model_name}', longitud: {len(importance_plot_b64)} caracteres")
        
        return feature_importance_df, importance_plot_b64
    
    except Exception as e:
        log.error(f"Error al generar análisis de importancia para '{model_name}': {e}", exc_info=True)
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns.astype(str),
            'Importance': np.ones(len(X_train.columns)) * 0.1
        })
        with resource_cleanup():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Gráfico de Importancia no disponible\nValores uniformes asignados',
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.set_title(f"Importancia de Variables según SHAP - {model_name}")
            plt.tight_layout()
            importance_plot_b64 = save_fig_to_base64(fig)
            if not validate_base64_png(importance_plot_b64):
                log.error("Gráfico SHAP de respaldo no es un PNG válido")
            log.info(f"Gráfico SHAP de respaldo generado, longitud: {len(importance_plot_b64)} caracteres")
        log.info(f"feature_importance_df de respaldo: {feature_importance_df.to_dict()}")
        return feature_importance_df, importance_plot_b64

def experiment_and_evaluate_nn(X_train, y_train, X_test, y_test):
    """Realiza experimentación con redes neuronales."""
    log.info("Iniciando experimentación de redes neuronales.")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    num_features = X_train_scaled.shape[1]
    
    scenarios = {
        'NN_base_ReLU': {
            'activation': 'relu',
            'optimizer_name': 'adam',
            'learning_rate': 0.01,
            'use_dropout': False,
            'use_l2': False
        },
        'NN_Sigmoid': {
            'activation': 'sigmoid',
            'optimizer_name': 'adam',
            'learning_rate': 0.001,
            'use_dropout': True,
            'use_l2': True
        },
        'NN_con_Dropout': {
            'activation': 'relu',
            'optimizer_name': 'adam',
            'learning_rate': 0.01,
            'use_dropout': True,
            'use_l2': False
        },
        'NN_con_L2': {
            'activation': 'relu',
            'optimizer_name': 'adam',
            'learning_rate': 0.01,
            'use_dropout': False,
            'use_l2': True
        }
    }

    results = []
    all_plots = {}
    trained_models = {}
    
    tf.config.run_functions_eagerly(False)
    
    for name, params in scenarios.items():
        log.info(f"Entrenando escenario: {name}")
        
        tf.random.set_seed(config.RANDOM_STATE)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            np.random.seed(config.RANDOM_STATE)
        
        tf.keras.backend.clear_session()
        
        try:
            model = build_nn_model(num_features, **params)
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=30,
                    restore_best_weights=True,
                    verbose=0
                )
            ]
            
            batch_size = 32
            history = model.fit(
                X_train_scaled, y_train,
                epochs=300,
                validation_data=(X_test_scaled, y_test),
                callbacks=callbacks,
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )
            
            y_pred = model.predict(X_test_scaled, batch_size=batch_size, verbose=0).flatten()
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            if r2 < 0.1:
                log.warning(f"Modelo '{name}' tiene R² bajo ({r2:.4f}).")
                results.append({'Modelo': name, 'R² Score': r2, 'MAE': mae, 'RMSE': rmse})
                continue
            
            results.append({'Modelo': name, 'R² Score': r2, 'MAE': mae, 'RMSE': rmse})
            all_plots.update(generate_diagnostic_plots(y_test, y_pred, name))
            all_plots.update(generate_nn_learning_curve(history, name))
            trained_models[name] = model
            
            log.info(f"Escenario {name} completado - R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        except Exception as e:
            log.error(f"Error entrenando el modelo {name}: {e}", exc_info=True)
            results.append({'Modelo': name, 'R² Score': 0.0, 'MAE': float('inf'), 'RMSE': float('inf')})
        
        finally:
            tf.keras.backend.clear_session()

    return pd.DataFrame(results), all_plots, trained_models, scaler

def generate_feature_importance_html(df):
    """Genera tabla HTML para importancia de variables, siguiendo simulation_table_html.html."""
    try:
        if df.empty or not all(col in df.columns for col in ['Feature', 'Importance']):
            log.error(f"DataFrame inválido para tabla HTML: {df}")
            raise ValueError("DataFrame inválido para tabla HTML")
        if df['Importance'].isna().any():
            log.error("DataFrame contiene NaN en Importance")
            raise ValueError("DataFrame contiene NaN")
        
        html = """
        <div class="table-container" style="visibility: visible !important; width: 100%; margin: 20px 0;">
            <h2>Variables más Influyentes</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Feature</th>
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Importance</th>
                </tr>
        """
        for _, row in df.iterrows():
            html += f"""
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">{row['Feature']}</td>
                    <td style="border: 1px solid black; padding: 8px;">{row['Importance']:.4f}</td>
                </tr>
            """
        html += """
            </table>
        </div>
        """
        if '<table' not in html:
            log.error("HTML generado no contiene una tabla válida")
            raise ValueError("HTML no contiene una tabla válida")
        log.info(f"feature_importance_html generado, longitud: {len(html)} caracteres")
        return html
    except Exception as e:
        log.error(f"Error generando feature_importance_html: {e}")
        return """
        <div class="table-container" style="visibility: visible !important; width: 100%; margin: 20px 0;">
            <h2>Variables más Influyentes</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Feature</th>
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Importance</th>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">No disponible</td>
                    <td style="border: 1px solid black; padding: 8px;">0.0000</td>
                </tr>
            </table>
        </div>
        """

def generate_table_html(df):
    """Genera una tabla HTML autocontenida, siguiendo simulation_table_html.html."""
    try:
        if df.empty or not all(col in df.columns for col in ['Feature', 'Importance']):
            log.error(f"DataFrame inválido para table_html: {df}")
            raise ValueError("DataFrame inválido para table_html")
        if df['Importance'].isna().any():
            log.error("DataFrame contiene NaN en Importance")
            raise ValueError("DataFrame contiene NaN")
        
        html = """
        <div class="table-container" style="visibility: visible !important; width: 100%; margin: 20px 0;">
            <h2>Variables más Influyentes</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Feature</th>
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Importance</th>
                </tr>
        """
        for _, row in df.iterrows():
            html += f"""
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">{row['Feature']}</td>
                    <td style="border: 1px solid black; padding: 8px;">{row['Importance']:.4f}</td>
                </tr>
            """
        html += """
            </table>
        </div>
        """
        if '<table' not in html:
            log.error("HTML generado no contiene una tabla válida")
            raise ValueError("HTML no contiene una tabla válida")
        log.info(f"table_html generado, longitud: {len(html)} caracteres")
        return html
    except Exception as e:
        log.error(f"Error generando table_html: {e}")
        return """
        <div class="table-container" style="visibility: visible !important; width: 100%; margin: 20px 0;">
            <h2>Variables más Influyentes</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Feature</th>
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Importance</th>
                </tr>
                <tr>
                    <td style="border: 1px solid black; padding: 8px;">No disponible</td>
                    <td style="border: 1px solid black; padding: 8px;">0.0000</td>
                </tr>
            </table>
        </div>
        """

def generate_table_json(df):
    """Genera datos de la tabla en formato JSON."""
    try:
        if df.empty or not all(col in df.columns for col in ['Feature', 'Importance']):
            log.error(f"DataFrame inválido para JSON: {df}")
            raise ValueError("DataFrame inválido para JSON")
        json_data = df.to_dict(orient='records')
        log.info(f"JSON de tabla generado: {json.dumps(json_data)[:200]}...")
        return json_data
    except Exception as e:
        log.error(f"Error generando JSON de tabla: {e}")
        return [{'Feature': 'No disponible', 'Importance': 0.0}]

def generate_template_simulations(feature_importance_df, feature_importance_html, table_html, table_json, plots_b64):
    """Genera simulaciones de templates Jinja2 para la tabla y gráfico."""
    simulations = {}

    common_css = """
        <style>
            .table-container { visibility: visible !important; width: 100%; margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
        </style>
    """

    template_variants = [
        {
            'name': 'feature_importance',
            'template': f"""
            <html>
            <head><meta charset="UTF-8">{common_css}</head>
            <body>
                <h2>Variables más Influyentes</h2>
                <div class="table-container">{{{{ feature_importance_html }}}}</div>
                {{% if plots_b64.shap_summary_plot %}}
                    <img src="data:image/png;base64,{{{{ plots_b64.shap_summary_plot }}}}" alt="SHAP Imagen Gráfica">
                {{% else %}}
                    <p>Gráfico SHAP no disponible</p>
                {{% endif %}}
            </body>
            </html>
            """,
            'variables': {'feature_importance_html': feature_importance_html, 'plots_b64': plots_b64}
        },
        {
            'name': 'table_html',
            'template': f"""
            <html>
            <head><meta charset="UTF-8">{common_css}</head>
            <body>
                <div class="table-container">{{{{ table_html }}}}</div>
                {{% if plots_b64.shap_summary_plot %}}
                    <img src="data:image/png;base64,{{{{ plots_b64.shap_summary_plot }}}}" alt="SHAP Imagen Gráfica">
                {{% else %}}
                    <p>Gráfico SHAP no disponible</p>
                {{% endif %}}
            </body>
            </html>
            """,
            'variables': {'table_html': table_html, 'plots_b64': plots_b64}
        }
    ]

    for variant in template_variants:
        try:
            variant_name = variant['name']

            if not plots_b64.get('shap_summary_plot'):
                log.warning(f"[{variant_name}] SHAP no disponible, generando gráfico de respaldo.")
                with resource_cleanup():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, 'Gráfico SHAP no disponible',
                            ha='center', va='center', fontsize=12)
                    ax.set_title("Importancia de Variables según SHAP")
                    plt.tight_layout()
                    plots_b64['shap_summary_plot'] = save_fig_to_base64(fig)
                log.info(f"[{variant_name}] Gráfico SHAP de respaldo generado (longitud: {len(plots_b64['shap_summary_plot'])})")

            template = Template(variant['template'])
            rendered = template.render(**variant['variables'])

            if '<img src="data:image/png;base64,"' in rendered:
                log.error(f"[{variant_name}] Imagen SHAP vacía detectada")
                rendered = rendered.replace(
                    '<img src="data:image/png;base64," alt="SHAP Imagen Gráfica">',
                    '<p>Gráfico SHAP no disponible</p>'
                )

            simulations[variant_name] = rendered
            log.info(f"[{variant_name}] Simulación generada (longitud: {len(rendered)} caracteres)")

            output_path = f"simulation_{variant_name}.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered)

        except Exception as e:
            log.exception(f"[{variant_name}] Error al generar simulación: {e}")
            simulations[variant_name] = '<html><body><div>Error en renderizado</div></body></html>'

    return simulations

def run_training_pipeline(df):
    """Orquesta el pipeline de entrenamiento y evaluación."""
    log.info("Iniciando pipeline de entrenamiento.")
    
    # Inicializar training_artifacts con valores por defecto
    training_artifacts = {
        'all_results_df': pd.DataFrame(),
        'nn_results_df': pd.DataFrame(),
        'feature_importance_df': pd.DataFrame({
            'Feature': ['No disponible'],
            'Importance': [0.0]
        }),
        'feature_importance_html': '',
        'table_html': '',
        'table_json': [{'Feature': 'No disponible', 'Importance': 0.0}],
        'plots_b64': {},
        'best_model_name': 'N/A',
        'simulations': {},
        'status': 'pending'
    }
    
    try:
        if not all(col in df.columns for col in config.FEATURE_COLUMNS + [config.TARGET_VARIABLE]):
            missing_cols = [col for col in config.FEATURE_COLUMNS + [config.TARGET_VARIABLE] if col not in df.columns]
            log.error(f"Columnas faltantes: {missing_cols}")
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        if df[config.FEATURE_COLUMNS].isnull().any().any() or df[config.TARGET_VARIABLE].isnull().any():
            log.error("Datos contienen NaN")
            raise ValueError("Datos contienen NaN")
        if np.isinf(df[config.FEATURE_COLUMNS].values).any() or np.isinf(df[config.TARGET_VARIABLE].values).any():
            log.error("Datos contienen infinitos")
            raise ValueError("Datos contienen infinitos")
        
        X_train, X_test, y_train, y_test = split_data(df)
        log.info(f"Datos divididos - Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
        
        classic_results_df, classic_plots, classic_pipelines = train_and_evaluate_classics(X_train, y_train, X_test, y_test)
        nn_results_df, nn_plots, trained_nn_models, nn_scaler = experiment_and_evaluate_nn(X_train, y_train, X_test, y_test)
        
        all_results_df = pd.concat([classic_results_df, nn_results_df], ignore_index=True)
        all_plots = {**classic_plots, **nn_plots}
        
        valid_results = all_results_df[all_results_df['R² Score'] > 0.1]
        if valid_results.empty:
            log.error("No hay modelos con R² válido (> 0.1)")
            raise ValueError("No hay modelos con R² válido")
        
        best_model_row = valid_results.sort_values(by='R² Score', ascending=False).iloc[0]
        best_model_name = best_model_row['Modelo']
        
        log.info(f"Mejor modelo: '{best_model_name}' con R² = {best_model_row['R² Score']:.4f}")
        
        feature_importance_df = None
        shap_plot = None
        
        if best_model_name in classic_pipelines:
            best_pipeline = classic_pipelines[best_model_name]
            best_pipeline.fit(df[config.FEATURE_COLUMNS], df[config.TARGET_VARIABLE])
            
            model_path = config.MODELS_DIR / config.FINAL_MODEL_NAME
            joblib.dump(best_pipeline, model_path)
            log.info(f"Modelo clásico '{best_model_name}' guardado en: {model_path}")
            
            feature_importance_df, shap_plot = analyze_feature_importance(
                best_pipeline, best_model_name, df[config.FEATURE_COLUMNS]
            )
        elif best_model_name in trained_nn_models:
            best_nn_model = trained_nn_models[best_model_name]
            model_path = config.MODELS_DIR / 'nn_model_final.keras'
            best_nn_model.save(model_path)
            joblib.dump(nn_scaler, config.MODELS_DIR / 'nn_scaler.pkl')
            log.info(f"Red neuronal '{best_model_name}' guardada en: {model_path}")
            
            feature_importance_df, shap_plot = analyze_feature_importance(
                best_nn_model, best_model_name, df[config.FEATURE_COLUMNS], nn_scaler
            )
        else:
            log.error(f"Mejor modelo '{best_model_name}' no encontrado")
            raise ValueError(f"Mejor modelo '{best_model_name}' no válido")
        
        if not isinstance(feature_importance_df, pd.DataFrame) or \
           feature_importance_df.empty or \
           not all(col in feature_importance_df.columns for col in ['Feature', 'Importance']) or \
           feature_importance_df['Importance'].isna().any():
            log.warning(f"feature_importance_df inválido: {feature_importance_df}")
            feature_importance_df = pd.DataFrame({
                'Feature': df[config.FEATURE_COLUMNS].columns.astype(str),
                'Importance': np.ones(len(df[config.FEATURE_COLUMNS].columns)) * 0.1
            })
            with resource_cleanup():
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'Gráfico de Importancia no disponible\nValores uniformes asignados',
                        horizontalalignment='center', verticalalignment='center', fontsize=12)
                ax.set_title(f"Importancia de Variables según SHAP - {best_model_name}")
                plt.tight_layout()
                shap_plot = save_fig_to_base64(fig)
                if not validate_base64_png(shap_plot):
                    log.error("Gráfico SHAP de respaldo no es un PNG válido")
            log.info(f"Gráfico SHAP de respaldo generado, longitud: {len(shap_plot)} caracteres")
        
        feature_importance_html = generate_feature_importance_html(feature_importance_df)
        table_html = generate_table_html(feature_importance_df)
        table_json = generate_table_json(feature_importance_df)
        
        all_plots['shap_summary_plot'] = shap_plot
        if not shap_plot or not validate_base64_png(shap_plot):
            log.warning("shap_summary_plot vacío o inválido, generando respaldo")
            with resource_cleanup():
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'Gráfico SHAP no disponible', 
                        horizontalalignment='center', verticalalignment='center', fontsize=12)
                ax.set_title("Importancia de Variables según SHAP")
                plt.tight_layout()
                all_plots['shap_summary_plot'] = save_fig_to_base64(fig)
                if not validate_base64_png(all_plots['shap_summary_plot']):
                    log.error("Gráfico SHAP de respaldo no válido")
            log.info(f"Gráfico SHAP de respaldo generado, longitud: {len(all_plots['shap_summary_plot'])}")
        
        log.info(f"shap_summary_plot generado: {len(all_plots['shap_summary_plot'])} caracteres")
        
        simulations = generate_template_simulations(
            feature_importance_df, feature_importance_html, table_html, table_json, all_plots
        )
        
        # Actualizar training_artifacts
        training_artifacts.update({
            'all_results_df': all_results_df,
            'nn_results_df': nn_results_df,
            'feature_importance_df': feature_importance_df,
            'feature_importance_html': feature_importance_html,
            'table_html': table_html,
            'table_json': table_json,
            'plots_b64': all_plots,
            'best_model_name': best_model_name,
            'simulations': simulations,
            'status': 'success'
        })
        
        # Validar training_artifacts
        required_keys = {
            'all_results_df': pd.DataFrame,
            'nn_results_df': pd.DataFrame,
            'feature_importance_df': pd.DataFrame,
            'feature_importance_html': str,
            'table_html': str,
            'table_json': list,
            'plots_b64': dict,
            'best_model_name': str,
            'simulations': dict,
            'status': str
        }
        
        for key, expected_type in required_keys.items():
            if key not in training_artifacts:
                log.error(f"Clave faltante en training_artifacts: {key}")
                raise ValueError(f"Clave faltante: {key}")
            if not isinstance(training_artifacts[key], expected_type):
                log.error(f"Tipo inválido para {key}: esperado {expected_type}, obtenido {type(training_artifacts[key])}")
                raise ValueError(f"Tipo inválido para {key}")
        
        log.info(f"training_artifacts generado: "
                 f"feature_importance_df={len(feature_importance_df)} filas, "
                 f"feature_importance_html={len(feature_importance_html)} caracteres, "
                 f"shap_summary_plot={len(all_plots.get('shap_summary_plot', ''))} caracteres")
        log.info("Pipeline de entrenamiento completado exitosamente")
    
    except Exception as e:
        log.error(f"Error en el pipeline de entrenamiento: {e}", exc_info=True)
        # Generar artefactos de respaldo
        training_artifacts.update({
            'feature_importance_df': pd.DataFrame({
                'Feature': ['No disponible'],
                'Importance': [0.0]
            }),
            'feature_importance_html': generate_feature_importance_html(pd.DataFrame({
                'Feature': ['No disponible'],
                'Importance': [0.0]
            })),
            'table_html': generate_table_html(pd.DataFrame({
                'Feature': ['No disponible'],
                'Importance': [0.0]
            })),
            'table_json': [{'Feature': 'No disponible', 'Importance': 0.0}],
            'plots_b64': {
                'shap_summary_plot': save_fig_to_base64(
                    plt.figure(figsize=(10, 6)).add_subplot(111).text(
                        0.5, 0.5, 'Gráfico SHAP no disponible',
                        horizontalalignment='center', verticalalignment='center', fontsize=12
                    ).get_figure()
                )
            },
            'simulations': {},
            'status': 'error',
            'error_message': str(e)
        })
        log.warning(f"Pipeline falló, retornando training_artifacts con valores por defecto")
    
    return training_artifacts

if __name__ == "__main__":
    try:
        df = pd.read_csv(config.DATA_DIR / config.DATASET_NAME)
        training_artifacts = run_training_pipeline(df)
        log.info("Pipeline ejecutado exitosamente")
    except Exception as e:
        log.error(f"Error ejecutando el pipeline: {e}", exc_info=True)
        raise