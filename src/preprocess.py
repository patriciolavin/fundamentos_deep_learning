# src/preprocess.py
import pandas as pd
import numpy as np
from utils.logger import log
from utils import config

def preprocess_data(df):
    """
    Realiza el preprocesamiento de datos, incluyendo manejo de valores nulos y transformaciones.
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos.
    Returns:
        pd.DataFrame: DataFrame preprocesado.
    """
    log.info("Iniciando preprocesamiento de datos.")
    
    # Validar columnas esperadas
    required_columns = config.FEATURE_COLUMNS + [config.TARGET_VARIABLE]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        log.error(f"Columnas faltantes en el DataFrame: {missing_cols}")
        raise ValueError(f"Columnas faltantes: {missing_cols}")
    
    # CORRECCIÓN: Validar que las columnas esperadas sean numéricas
    non_numeric_cols = [
        col for col in required_columns 
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        log.error(f"Columnas no numéricas detectadas: {non_numeric_cols}")
        raise ValueError(f"Columnas no numéricas: {non_numeric_cols}")
    
    # Copia del DataFrame para evitar modificaciones no deseadas
    df_processed = df.copy()
    
    # Manejo de valores nulos (imputación con la mediana para columnas numéricas)
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            median_value = df_processed[col].median()
            df_processed[col].fillna(median_value, inplace=True)
            log.info(f"Imputados valores nulos en {col} con la mediana: {median_value}")
    
    # Validar que no hay valores nulos
    if df_processed.isnull().any().any():
        log.error("Valores nulos detectados después del preprocesamiento.")
        raise ValueError("Valores nulos detectados después del preprocesamiento")
    
    # CORRECCIÓN: Verificar valores infinitos solo en columnas numéricas
    numeric_df = df_processed[numeric_cols]
    if np.isinf(numeric_df.values).any():
        inf_cols = numeric_df.columns[np.isinf(numeric_df.values).any(axis=0)].tolist()
        log.error(f"Valores infinitos detectados en columnas: {inf_cols}")
        raise ValueError(f"Valores infinitos detectados en columnas: {inf_cols}")
    
    log.info("Preprocesamiento de datos finalizado.")
    return df_processed