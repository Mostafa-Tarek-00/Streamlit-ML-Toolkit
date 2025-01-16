import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    label_encoders = {}

    st.write(df.dtypes)

    for column_name in df.columns:
        if df[column_name].dtype == 'object':
            converted_column = pd.to_numeric(df[column_name], errors='coerce')
            if not converted_column.isna().all():
                df[column_name] = converted_column
                st.write(f"The {column_name} column has been converted to numeric to solve problems like '5' being converted to {5}")
                
    def calculate_null_percentage(column):
        total_rows = len(column)
        null_count = column.isnull().sum()
        return (null_count / total_rows) * 100

    def fill_nulls(column, column_name):
        if pd.api.types.is_numeric_dtype(column):
            median_value = column.median()
            column.fillna(median_value, inplace=True)
            st.write(f"The {column_name} column has been filled with the median value for nulls")
        else:
            mode_value = column.mode()[0]
            column.fillna(mode_value, inplace=True)
            st.write(f"The {column_name} column has been filled with the mode value for nulls")

    null_percentages = [(column_name, calculate_null_percentage(df[column_name])) for column_name in df.columns]
    null_percentages.sort(key=lambda x: x[1], reverse=True)

    for column_name, null_percentage in null_percentages:
        column = df[column_name]
        st.write(f"The {column_name} column has {df[column_name].isnull().sum()} nulls and its percentage is {null_percentage}%")

        if null_percentage <= 30:
            df = df[~column.isnull()]
            st.write(f"The {column_name} column has been dropped due to null values")
        else:
            fill_nulls(column, column_name)

    for column_name in df.columns:
        if df[column_name].dtype == 'object':
            label_encoder = LabelEncoder()
            encoded_values = label_encoder.fit_transform(df[column_name])
            df[column_name] = encoded_values
            label_encoders[column_name] = label_encoder
            st.write(f"The {column_name} column has been label-encoded")
            df[column_name] = df[column_name].dropna()

    return df, label_encoders
