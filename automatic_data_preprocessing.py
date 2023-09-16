import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
        label_encoders = {}

        st.write(df.dtypes)

        def calculate_null_percentage(column):
            total_rows = len(column)
            null_count = column.isnull().sum()
            return (null_count / total_rows) * 100

        def fill_nulls(column):
            if pd.api.types.is_numeric_dtype(column):
                median_value = column.median()
                df[column_name].fillna(median_value, inplace=True)
                st.write(f"The {column_name} column is median nulls")
            else:
                mode_value = column.mode()[0]
                df[column_name].fillna(mode_value, inplace=True)
                st.write(f"The {column_name} column is mode nulls")

        for column_name in df.columns:
            if df[column_name].dtype == 'object':
                converted_column = pd.to_numeric(df[column_name], errors='coerce')
                if not converted_column.isna().all():
                    df[column_name] = converted_column
                    st.write(f"This column {column_name} has been converted to numirec to solve problem like '5' convert it to {5}")
                else:
                    label_encoder = LabelEncoder()
                    encoded_values = label_encoder.fit_transform(df[column_name])
                    df[column_name] = encoded_values
                    label_encoders[column_name] = label_encoder
                    st.write(f"The {column_name} column  has been labeled encoded")

        null_percentages = [(column_name, calculate_null_percentage(df[column_name])) for column_name in df.columns]
        null_percentages.sort(key=lambda x: x[1], reverse=True)
        
        for column_name, null_percentage in null_percentages:
            column = df[column_name]
            st.write(f"The {column_name} column  has {df[column_name].isnull().sum()} Nulls and it's percentage is {null_percentages}")

            if null_percentage <= 30:
                df = df[~column.isnull()]
                st.write(f"The {column_name} column is Dropped nulls")
            else:
                fill_nulls(column)

        return df, label_encoders
