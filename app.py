import streamlit as st
import json
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="Evaluador de Clasificación Multi-Etiqueta", layout="wide")

st.title("Evaluador de Clasificación Multi-Etiqueta")
st.write("Suba los valores reales y los valores predichos para calcular las métricas de precision y recall.")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    actual_file = st.file_uploader("Subir Valores Reales (JSON)", type=["txt", "json"])
with col2:
    predicted_file = st.file_uploader("Subir Valores Predichos (JSON)", type=["txt", "json"])

def calculate_metrics(actual_dict, predicted_dict):
    """Calculate precision and recall for each filename and global averages."""
    results = []
    
    # Find all unique filenames across both dictionaries
    all_filenames = set(actual_dict.keys()) | set(predicted_dict.keys())
    
    for filename in all_filenames:
        # Get labels for the current filename
        actual_labels = set(actual_dict.get(filename, []))
        predicted_labels = set(predicted_dict.get(filename, []))
        
        # Calculate metrics with user-friendly terminology
        found_correct = len(actual_labels & predicted_labels)  # Intersection
        found_incorrect = len(predicted_labels - actual_labels)  # In predicted but not in actual
        not_found = len(actual_labels - predicted_labels)  # In actual but not in predicted
        
        # Calculate precision and recall
        precision = found_correct / (found_correct + found_incorrect) if (found_correct + found_incorrect) > 0 else 0
        recall = found_correct / (found_correct + not_found) if (found_correct + not_found) > 0 else 0
        
        # Store results with user-friendly terminology
        results.append({
            'Archivo': filename,
            'Categorias Verdad': list(actual_labels),
            'Categorias LLM': list(predicted_labels),
            'Total': len(actual_labels),  # Total number of labels in ground truth
            'Correctas': found_correct,
            'Incorrectas': found_incorrect,
            'No Encontradas': not_found,
            'Precision': precision,
            'Recall': recall
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate global metrics (average of per-filename metrics)
    global_precision = results_df['Precision'].mean()
    global_recall = results_df['Recall'].mean()
    
    return results_df, global_precision, global_recall

# Process files when both are uploaded
if actual_file and predicted_file:
    try:
        # Read and parse JSON data
        actual_content = actual_file.read().decode('utf-8')
        predicted_content = predicted_file.read().decode('utf-8')
        
        actual_dict = json.loads(actual_content)
        predicted_dict = json.loads(predicted_content)
        
        # Calculate metrics
        results_df, global_precision, global_recall = calculate_metrics(actual_dict, predicted_dict)
        
        # Display global metrics
        st.header("Métricas Globales")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Recall Global", f"{global_recall:.4f}")
        with metric_col2:
            st.metric("Precision Global", f"{global_precision:.4f}")
        
        # Display per-filename metrics
        st.header("Métricas por Archivo")
        
        # Create a more compact view of the results
        display_df = results_df[['Archivo', 'Recall', 'Precision']].copy()
        # Ensure the display DataFrame maintains the same sorting
        st.dataframe(display_df, use_container_width=True)
        
        # Detailed view (expandable)
        with st.expander("Mostrar Análisis Detallado"):
            st.dataframe(results_df, use_container_width=True)
            
            # Download button for results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Descargar Resultados como CSV",
                data=csv,
                file_name="classification_metrics.csv",
                mime="text/csv"
            )
        
        # Add explanation of metrics (permanently visible)
        st.subheader("Entendiendo las Métricas")
        st.markdown("""
        * **Total**: Número de etiquetas en los valores reales
        * **Correctas**: Etiquetas predichas correctamente
        * **Incorrectas**: Etiquetas predichas incorrectamente
        * **No Encontradas**: Etiquetas que debieron ser predichas pero no lo fueron
        * **Recall**: Correctas / (Correctas + No Encontradas)
        * **Precision**: Correctas / (Correctas + Incorrectas)
        """)
            
    except Exception as e:
        st.error(f"Error al procesar archivos: {str(e)}")
        st.write("Por favor, asegúrese de que los archivos subidos contienen datos JSON válidos con el formato esperado.")