# main.py
from utils.logger import log
from src import data_processing, eda, train_model, reporting
from utils import config  # CORRECCIÓN: Importar config para validaciones

def main():
    """
    Función principal que orquesta todo el pipeline de ML.
    """
    log.info("================================================")
    log.info("INICIANDO PIPELINE DE PREDICCIÓN DE NATALIDAD")
    log.info("================================================")

    try:
        # Fase 1: Carga y Limpieza de Datos
        log.info("--- Fase 1: Carga y Limpieza de Datos ---")
        df = data_processing.load_data()
        # CORRECCIÓN: Validar que el DataFrame contiene las columnas esperadas
        required_columns = config.FEATURE_COLUMNS + [config.TARGET_VARIABLE]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            log.error(f"Columnas faltantes en el dataset: {missing_cols}")
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        cleaned_df = data_processing.clean_data(df)
        
        # Fase 2: Análisis Exploratorio de Datos (EDA)
        log.info("--- Fase 2: Análisis Exploratorio de Datos ---")
        eda_results = eda.perform_eda(cleaned_df)

        # Fase 3: Entrenamiento, Evaluación y Guardado de Modelos
        log.info("--- Fase 3: Pipeline de Entrenamiento Completo ---")
        training_artifacts = train_model.run_training_pipeline(cleaned_df)

        # Fase 4: Generación de Reporte
        log.info("--- Fase 4: Generación de Reporte ---")
        reporting.generate_report(eda_results, training_artifacts)

        log.info("=" * 71)
        log.info("✅ EVALUACIÓN FINAL MÓDULO 8 COMPLETADA EXITOSAMENTE ".center(70, "="))
        log.info("=" * 71)

    except Exception as e:
        log.error(f"El pipeline falló con el siguiente error: {e}", exc_info=True)
        log.info("================================================")
        log.info("PIPELINE FINALIZADO CON ERRORES")
        log.info("================================================")
        raise  # CORRECCIÓN: Re-lanzar la excepción para depuración

if __name__ == '__main__':
    main()