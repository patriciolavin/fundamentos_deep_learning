# src/reporting.py
import pandas as pd
import os
import base64
from jinja2 import Environment, FileSystemLoader
from utils import config
from utils.logger import log
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns

def validate_base64_png(image_data):
    """
    Valida si un string base64 o una ruta de archivo representa una imagen PNG válida.
    """
    try:
        if isinstance(image_data, str) and os.path.exists(image_data):
            with open(image_data, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        decoded = base64.b64decode(image_data)
        if decoded.startswith(b'\x89PNG\r\n\x1a\n'):
            log.debug("Imagen base64 es un PNG válido")
            return True
        else:
            log.error("Imagen base64 no es un PNG válido")
            return False
    except Exception as e:
        log.error(f"Error validando base64 PNG: {e}")
        return False

def create_placeholder_image(message):
    """
    Genera una imagen de placeholder con un mensaje de texto.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, message, fontsize=12, horizontalalignment='center', verticalalignment='center', wrap=True)
        ax.set_axis_off()
        plt.title("Placeholder")
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        if not img_b64:
            log.error("Imagen placeholder base64 vacía")
            raise ValueError("Imagen placeholder base64 vacía")
        log.debug(f"Placeholder generado, longitud: {len(img_b64)} caracteres")
        return img_b64
    except Exception as e:
        log.error(f"Error generando placeholder: {e}")
        raise

def validate_html_table(html_content):
    """
    Valida que el contenido HTML contenga una tabla válida.
    """
    if not isinstance(html_content, str) or not html_content.strip():
        log.error("Contenido HTML vacío")
        return False
    if '<table' not in html_content:
        log.error("No se encontró etiqueta <table> en HTML")
        return False
    if '<tr' not in html_content:
        log.error("No se encontraron filas en la tabla HTML")
        return False
    return True

def generate_report(eda_results, training_artifacts):
    """
    Genera el reporte HTML final, asegurando que el contexto para Jinja2 esté siempre bien definido y completo.
    """
    log.info("Iniciando la generación del reporte HTML.")
    
    try:
        env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR))
        template = env.get_template('template.html')
    except Exception as e:
        log.error(f"Error fatal al cargar la plantilla 'template.html': {e}")
        raise

    # Extraer artefactos de forma segura
    all_results_df = training_artifacts.get('all_results_df', pd.DataFrame())
    # Usar 'nn_results_df' en lugar de 'nn_comparison_df' para la tabla de redes neuronales
    nn_comparison_df = training_artifacts.get('nn_results_df', pd.DataFrame())
    feature_importance_html = training_artifacts.get('feature_importance_html', None)
    plots_b64 = training_artifacts.get('plots_b64', {})
    eda_images = eda_results.get('images_b64', {})

    # Validar nn_comparison_df
    expected_columns = ['Modelo', 'R² Score', 'MAE', 'RMSE']
    if not nn_comparison_df.empty and not all(col in nn_comparison_df.columns for col in expected_columns):
        log.error(f"nn_comparison_df no contiene las columnas esperadas: {nn_comparison_df.columns}")
        nn_comparison_df = pd.DataFrame(columns=expected_columns)
        log.warning("Usando DataFrame vacío como respaldo para nn_comparison_df")

    log.info(f"nn_comparison_df contiene {len(nn_comparison_df)} filas")

    # Validar feature_importance_html exhaustivamente
    if not validate_html_table(feature_importance_html):
        log.warning("feature_importance_html inválido o vacío, generando tabla de respaldo")
        feature_importance_df = training_artifacts.get('feature_importance_df', pd.DataFrame())
        if not isinstance(feature_importance_df, pd.DataFrame) or feature_importance_df.empty:
            log.warning("feature_importance_df inválido o vacío, usando valores por defecto")
            feature_importance_df = pd.DataFrame({
                'Feature': ['No disponible'],
                'Importance': [0.0]
            })
        html = """
        <div class="table-container" style="visibility: visible !important; width: 100%; margin: 20px 0;">
            <h2>Variables más Influyentes</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Feature</th>
                    <th style="border: 1px solid black; padding: 8px; text-align: left;">Importance</th>
                </tr>
        """
        for _, row in feature_importance_df.iterrows():
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
        feature_importance_html = html
        log.info(f"Tabla de respaldo generada para feature_importance_html, longitud: {len(feature_importance_html)} caracteres")

    # Validar shap_summary_plot de forma robusta
    shap_plot = plots_b64.get('shap_summary_plot')
    if not shap_plot or not isinstance(shap_plot, str) or not validate_base64_png(shap_plot):
        log.warning("shap_summary_plot no válido, generando placeholder")
        feature_importance_df = training_artifacts.get('feature_importance_df', pd.DataFrame())
        if isinstance(feature_importance_df, pd.DataFrame) and not feature_importance_df.empty:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
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
                ax.set_title("Importancia de Variables según SHAP")
                ax.set_xlabel('Importancia Normalizada')
                ax.set_ylabel('Variable')
                plt.tight_layout()
                shap_plot = create_placeholder_image("")  # Generar gráfico real en lugar de placeholder
                if not validate_base64_png(shap_plot):
                    log.error("Gráfico SHAP regenerado no es un PNG válido")
                    shap_plot = create_placeholder_image("Gráfico SHAP no disponible")
                log.info(f"Gráfico SHAP regenerado, longitud: {len(shap_plot)} caracteres")
            except Exception as e:
                log.error(f"Error regenerando gráfico SHAP: {e}")
                shap_plot = create_placeholder_image("Gráfico SHAP no disponible")
                log.info(f"Placeholder generado para shap_summary_plot, longitud: {len(shap_plot)} caracteres")
        else:
            shap_plot = create_placeholder_image("Gráfico SHAP no disponible")
            log.info(f"Placeholder generado para shap_summary_plot, longitud: {len(shap_plot)} caracteres")
        plots_b64['shap_summary_plot'] = shap_plot

    # Construir encoded_images con gráficos de EDA y entrenamiento
    encoded_images = {}
    required_eda_plots = ['distribucion_variables', 'heatmap_correlacion']
    for plot in required_eda_plots:
        if plot in eda_images and eda_images[plot]:
            encoded_images[plot] = eda_images[plot]
        else:
            log.warning(f"Gráfico '{plot}' no disponible en eda_results, generando placeholder")
            encoded_images[plot] = create_placeholder_image(f"Gráfico '{plot}' no disponible")
            log.info(f"Placeholder generado para {plot}, longitud: {len(encoded_images[plot])} caracteres")

    # Agrupar gráficos de modelos clásicos
    classic_plots_structured = {}
    classic_model_names = all_results_df[~all_results_df['Modelo'].str.contains('NN', case=False)]['Modelo'].unique()
    for model_name in classic_model_names:
        classic_plots_structured[model_name] = {
            'predicciones': plots_b64.get(f'predicciones_{model_name}'),
            'residuos': plots_b64.get(f'residuos_{model_name}')
        }

    # Agrupar gráficos de redes neuronales
    nn_plots_structured = {}
    nn_model_names = all_results_df[all_results_df['Modelo'].str.contains('NN', case=False)]['Modelo'].unique()
    for model_name in nn_model_names:
        nn_plots_structured[model_name] = {
            'predicciones': plots_b64.get(f'predicciones_{model_name}'),
            'residuos': plots_b64.get(f'residuos_{model_name}'),
            'curva_aprendizaje': plots_b64.get(f'curva_aprendizaje_{model_name}')
        }

    # Construir contexto
    context = {
        'eda_results': eda_results,
        'model_results_html': all_results_df.to_html(classes=['table', 'table-striped'], index=False, float_format="%.4f"),
        'nn_comparison_html': nn_comparison_df.to_html(classes=['table', 'table-striped'], index=False, float_format="%.4f"),
        'feature_importance_html': feature_importance_html,
        'classic_plots': classic_plots_structured,
        'nn_plots': nn_plots_structured,
        'best_model_name': training_artifacts.get('best_model_name', 'N/A'),
        'encoded_images': encoded_images,
        'plots_b64': plots_b64,
    }
    
    log.info(f"Contexto preparado para renderizado: "
             f"feature_importance_html={len(feature_importance_html)} caracteres, "
             f"shap_summary_plot={'presente' if 'shap_summary_plot' in plots_b64 else 'ausente'}, "
             f"nn_comparison_html={len(nn_comparison_df.to_html())} caracteres")
    
    try:
        report_path = config.REPORTS_DIR / config.REPORT_NAME
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(template.render(context))
        log.info(f"Reporte final generado exitosamente en: {report_path}")
    except Exception as e:
        log.error(f"Error al renderizar la plantilla HTML: {e}", exc_info=True)
        raise