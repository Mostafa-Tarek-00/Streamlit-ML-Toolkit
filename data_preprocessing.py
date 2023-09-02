import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
        label_encoders = {}

        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            st.error("Unsupported file format. Please select a CSV or Excel file.")
            return None, None

        def calculate_null_percentage(column):
            total_rows = len(column)
            null_count = column.isnull().sum()
            return (null_count / total_rows) * 100

        def split_nulls(column):
            is_numeric = column.apply(lambda x: pd.api.types.is_numeric_dtype(x))
            numeric_nulls = column[is_numeric]
            string_nulls = column[~is_numeric]
            return numeric_nulls, string_nulls

        def fill_nulls(column):
            if pd.api.types.is_numeric_dtype(column):
                median_value = column.median()
                column.fillna(median_value, inplace=True)
            else:
                mode_value = column.mode()[0]
                column.fillna(mode_value, inplace=True)

        for column_name in df.columns:
            if df[column_name].dtype == 'object':
                converted_column = pd.to_numeric(df[column_name], errors='coerce')
                if not converted_column.isna().all():
                    df[column_name] = converted_column
                else:
                    label_encoder = LabelEncoder()
                    encoded_values = label_encoder.fit_transform(df[column_name])
                    df[column_name] = encoded_values
                    label_encoders[column_name] = label_encoder

        null_percentages = [(column_name, calculate_null_percentage(df[column_name])) for column_name in df.columns]
        null_percentages.sort(key=lambda x: x[1], reverse=True)

        for column_name, null_percentage in null_percentages:
            column = df[column_name]

            if null_percentage <= 30:
                df = df[~column.isnull()]
            else:
                fill_nulls(column)

        return df, label_encoders
