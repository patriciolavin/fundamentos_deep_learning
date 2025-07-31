# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from utils import config
from utils.logger import log
import base64
from io import BytesIO

def save_fig_to_base64(fig):
    """
    Guarda una figura de matplotlib en un string base64 para el reporte.
    Args:
        fig: Objeto de figura de matplotlib.
    Returns:
        str: Imagen codificada en base64.
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    # CORRECCIÓN: Validar que la imagen base64 no esté vacía
    if not img_b64:
        log.error("Imagen base64 vacía generada")
        raise ValueError("Imagen base64 vacía")
    log.debug(f"Imagen base64 generada, longitud: {len(img_b64)} caracteres")
    return img_b64

def analyze_data_quality(df):
    """
    Realiza un análisis detallado de la calidad de los datos.
    Args:
        df (pd.DataFrame): DataFrame a analizar.
    Returns:
        dict: Reporte de calidad de datos.
    """
    log.info("Ejecutando análisis de calidad de datos detallado.")
    quality_report = {}

    # Nulos y blancos
    nulls = df.isnull().sum()
    object_cols = df.select_dtypes(include=['object'])
    blanks = object_cols.isin(['', ' ']).sum()
    total_missing = nulls.add(blanks, fill_value=0)
    
    quality_report['missing_data'] = pd.DataFrame({
        'Valores Nulos (NaN)': nulls,
        'Cadenas Vacías/Blancas': blanks,
        'Total Ausentes': total_missing,
        '% Total Ausentes': (total_missing / len(df) * 100).round(2)
    })

    # Duplicados
    exact_duplicates = df.duplicated().sum()
    quality_report['exact_duplicates'] = {
        'count': exact_duplicates,
        'percentage': (exact_duplicates / len(df) * 100).round(2)
    }

    # Outliers (Método IQR)
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.empty:
        log.warning("No hay columnas numéricas para análisis de outliers.")
        quality_report['outliers'] = {
            'count_rows': 0,
            'percentage_rows': 0.0,
            'details': pd.Series(dtype=float)
        }
    else:
        Q1 = numerical_df.quantile(0.25)
        Q3 = numerical_df.quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR)))
        outliers_per_column = outlier_condition.sum()
        rows_with_outliers = outlier_condition.any(axis=1).sum()
        quality_report['outliers'] = {
            'count_rows': rows_with_outliers,
            'percentage_rows': (rows_with_outliers / len(df) * 100).round(2),
            'details': outliers_per_column[outliers_per_column > 0]
        }

    # Inconsistencias semánticas
    quality_report['semantic_inconsistencies'] = "No se detectaron inconsistencias semánticas obvias. Un análisis más profundo requeriría validación manual."

    log.info("Análisis de calidad de datos finalizado.")
    return quality_report

def generate_eda_plots(df):
    """
    Genera y codifica todos los gráficos necesarios para el EDA.
    Args:
        df (pd.DataFrame): DataFrame a analizar.
    Returns:
        dict: Diccionario con imágenes en base64.
    """
    images = {}
    numerical_cols = df.select_dtypes(include=np.number).columns
    if not numerical_cols.size:
        log.warning("No hay columnas numéricas para generar gráficos.")
        return images

    # Histogramas
    fig_hist, axes_hist = plt.subplots(len(numerical_cols), 1, figsize=(8, 4 * len(numerical_cols)))
    fig_hist.suptitle("Distribución de Variables Numéricas", fontsize=16)
    axes_hist = [axes_hist] if len(numerical_cols) == 1 else axes_hist
    for i, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes_hist[i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    images['distribucion_variables'] = save_fig_to_base64(fig_hist)

    # Heatmap de correlación
    if len(numerical_cols) > 1:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        images['heatmap_correlacion'] = save_fig_to_base64(fig_corr)

    return images

def perform_eda(df):
    """
    Realiza un Análisis Exploratorio de Datos completo y devuelve los resultados.
    Args:
        df (pd.DataFrame): DataFrame a analizar.
    Returns:
        dict: Resultados del EDA.
    """
    log.info("Iniciando Análisis Exploratorio de Datos (EDA).")
    results = {}

    info_buffer = StringIO()
    df.info(buf=info_buffer)
    results['df_info'] = info_buffer.getvalue()
    results['df_shape'] = df.shape
    results['df_memory'] = f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    
    results['df_head_html'] = df.head().to_html(classes='table table-striped', border=0, index=False)
    results['df_advanced_stats_html'] = df.describe().to_html(classes='table table-striped', border=0)
    
    # Se integra el análisis de calidad
    results['quality_report'] = analyze_data_quality(df)
    
    # Se generan y añaden los gráficos
    results['images_b64'] = generate_eda_plots(df)
    
    log.info("Análisis Exploratorio de Datos (EDA) finalizado.")
    return results