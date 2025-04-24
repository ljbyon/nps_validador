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
            'Actual Labels': list(actual_labels),
            'Predicted Labels': list(predicted_labels),
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
            st.metric("Global Recall", f"{global_recall:.4f}")
        with metric_col2:
            st.metric("Global Precision", f"{global_precision:.4f}")
        
        # Display per-filename metrics
        st.header("Per-Filename Metrics")
        
        # Create a more compact view of the results
        display_df = results_df[['Filename', 'Recall', 'Precision']].copy()
        # Ensure the display DataFrame maintains the same sorting
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
        
        # Add explanation of metrics (permanently visible)
        st.subheader("Understanding the Metrics")
        st.markdown("""
        * **Total Labels**: Number of labels in the ground truth
        * **Found Correct**: Labels that were correctly predicted
        * **Found Incorrect**: Labels that were incorrectly predicted
        * **Not Found**: Labels that should have been predicted but weren't
        * **Recall**: Found Correct / (Found Correct + Not Found)
        * **Precision**: Found Correct / (Found Correct + Found Incorrect)
        """)
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        st.write("Please ensure the uploaded files contain valid JSON data with the expected format.")