from pycaret.regression import *
import pandas as pd
import streamlit as st

def auto_pycaret(processed_df, target_column):

    # Set up the Streamlit app title and header
    st.title('Model Comparison with PyCaret')
    st.header('Comparison of Regression Models')

    # Set up PyCaret
    s = setup(data=processed_df, target=target_column, session_id=123)

    # Compare models
    models = compare_models()

    # Get you the results in a pandas dataframe (results object)
    results = pull()

    # Display the table of compared models
    st.subheader('Comparison of Regression Models')
    st.table(results)

    # Display the best model separately
    st.subheader('Best Model Based on R2')
    st.subheader(f"Best Model Based on Accuracy is {results['Model'][0]} with {results['R2'][0]}")
