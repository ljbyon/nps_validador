import streamlit as st
import json
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="Multi-Label Classification Evaluator", layout="wide")

st.title("Multi-Label Classification Evaluator")
st.write("Upload actual values and predicted values to calculate precision and recall metrics.")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    actual_file = st.file_uploader("Upload Actual Values (JSON)", type=["txt", "json"])
with col2:
    predicted_file = st.file_uploader("Upload Predicted Values (JSON)", type=["txt", "json"])

def calculate_metrics(actual_dict, predicted_dict):
    """Calculate precision and recall for each filename and global averages."""
    results = []
    
    # Find all unique filenames across both dictionaries
    all_filenames = set(actual_dict.keys()) | set(predicted_dict.keys())
    
    for filename in all_filenames:
        # Get labels for the current filename
        actual_labels = set(actual_dict.get(filename, []))
        predicted_labels = set(predicted_dict.get(filename, []))
        
        # Calculate metrics
        true_positives = len(actual_labels & predicted_labels)  # Intersection
        false_positives = len(predicted_labels - actual_labels)  # In predicted but not in actual
        false_negatives = len(actual_labels - predicted_labels)  # In actual but not in predicted
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Store results
        results.append({
            'Filename': filename,
            'Actual Labels': list(actual_labels),
            'Predicted Labels': list(predicted_labels),
            'True Positives': true_positives,
            'False Positives': false_positives,
            'False Negatives': false_negatives,
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
        st.header("Global Metrics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Global Precision", f"{global_precision:.4f}")
        with metric_col2:
            st.metric("Global Recall", f"{global_recall:.4f}")
        
        # Display per-filename metrics
        st.header("Per-Filename Metrics")
        
        # Create a more compact view of the results
        display_df = results_df[['Filename', 'Precision', 'Recall']].copy()
        st.dataframe(display_df, use_container_width=True)
        
        # Detailed view (expandable)
        with st.expander("Show Detailed Analysis"):
            st.dataframe(results_df, use_container_width=True)
            
            # Download button for results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="classification_metrics.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.write("Please ensure the uploaded files contain valid JSON data with the expected format.")

# Show sample format
with st.expander("Expected Input Format"):
    st.code('''{
    "E-927337.mp3": [
        "Calidad de productos"
    ],
    "E-927379.mp3": [
        "Calidad de productos",
        "Descripcion de producto"
    ],
    "E-928420.mp3": [
        "Opciones de pago",
        "Stock de productos"
    ]
}''', language="json")