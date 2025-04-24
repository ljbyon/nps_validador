def normalize_text(text):
    """Convert text to lowercase and remove accents."""
    if not isinstance(text, str):
        return text
    # Convert to lowercase
    text = text.lower()
    # Remove accents (normalize to NFKD form and keep only ASCII chars)
    text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                  if not unicodedata.combining(c))
    return text

def normalize_data(data_dict):
    """Normalize all text in the data dictionary (keys and values)."""
    normalized_dict = {}
    for key, values in data_dict.items():
        # Normalize the key (filename)
        normalized_key = normalize_text(key)
        # Normalize each value in the list
        normalized_values = [normalize_text(value) for value in values]
        normalized_dict[normalized_key] = normalized_values
    return normalized_dict

import streamlit as st
import json
import pandas as pd
import numpy as np
from io import StringIO
import unicodedata
import re

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
    
    # Identify filenames that will be excluded (in actual but not in predicted)
    all_actual_filenames = set(actual_dict.keys())
    all_predicted_filenames = set(predicted_dict.keys())
    excluded_filenames = all_actual_filenames - all_predicted_filenames
    
    # Only evaluate filenames that exist in the predicted JSON
    evaluation_filenames = all_predicted_filenames
    
    for filename in evaluation_filenames:
        # Get labels for the current filename
        actual_labels = set(actual_dict.get(filename, []))
        predicted_labels = set(predicted_dict.get(filename, []))
        
        # Ensure all label strings are normalized
        actual_labels = {normalize_text(label) for label in actual_labels}
        predicted_labels = {normalize_text(label) for label in predicted_labels}
        
        # Calculate metrics with user-friendly terminology
        found_correct = len(actual_labels & predicted_labels)  # Intersection
        found_incorrect = len(predicted_labels - actual_labels)  # In predicted but not in actual
        not_found = len(actual_labels - predicted_labels)  # In actual but not in predicted
        
        # Calculate precision and recall
        precision = found_correct / (found_correct + found_incorrect) if (found_correct + found_incorrect) > 0 else 0
        recall = found_correct / (found_correct + not_found) if (found_correct + not_found) > 0 else 0
        
        # Store results with user-friendly terminology
        results.append({
            'Filename': filename,
            'Actual Labels': [normalize_text(label) for label in actual_labels],
            'Predicted Labels': [normalize_text(label) for label in predicted_labels],
            'Total Labels': len(actual_labels),  # Total number of labels in ground truth
            'Found Correct': found_correct,
            'Found Incorrect': found_incorrect,
            'Not Found': not_found,
            'Precision': precision,
            'Recall': recall
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate global metrics (average of per-filename metrics)
    global_precision = results_df['Precision'].mean()
    global_recall = results_df['Recall'].mean()
    
    return results_df, global_precision, global_recall, excluded_filenames

# Process files when both are uploaded
if actual_file and predicted_file:
    try:
        # Read and parse JSON data
        actual_content = actual_file.read().decode('utf-8')
        predicted_content = predicted_file.read().decode('utf-8')
        
        actual_dict = json.loads(actual_content)
        predicted_dict = json.loads(predicted_content)
        
        # Normalize text (to lowercase and remove accents)
        actual_dict = normalize_data(actual_dict)
        predicted_dict = normalize_data(predicted_dict)
        
        # Calculate metrics
        results_df, global_precision, global_recall, excluded_filenames = calculate_metrics(actual_dict, predicted_dict)
        
        # Display global metrics
        st.header("Métricas Globales")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Recall Global", f"{global_recall:.4f}")
        with metric_col2:
            st.metric("Precision Global", f"{global_precision:.4f}")
            
        # Display excluded filenames
        if excluded_filenames:
            st.warning(f"**Archivos Excluidos de la Evaluación:** {len(excluded_filenames)} archivos presentes en los valores reales pero no en las predicciones no fueron evaluados.")
            with st.expander("Ver Archivos Excluidos"):
                st.write(", ".join(sorted(excluded_filenames)) if excluded_filenames else "Ninguno")
        
        # Display per-filename metrics
        st.header("Métricas por Archivo")
        
        # Create a more compact view of the results
        display_df = results_df[['Filename', 'Recall', 'Precision']].copy()
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
        * **Total Labels**: Número de etiquetas en los valores reales
        * **Found Correct**: Etiquetas predichas correctamente
        * **Found Incorrect**: Etiquetas predichas incorrectamente
        * **Not Found**: Etiquetas que debieron ser predichas pero no lo fueron
        * **Recall**: Found Correct / (Found Correct + Not Found)
        * **Precision**: Found Correct / (Found Correct + Found Incorrect)
        """)
            
    except Exception as e:
        st.error(f"Error al procesar archivos: {str(e)}")
        st.write("Por favor, asegúrese de que los archivos subidos contienen datos JSON válidos con el formato esperado.")