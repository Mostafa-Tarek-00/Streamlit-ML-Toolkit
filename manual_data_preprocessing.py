import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def preprocess_data(df):
    label_encoders = {}

    categorical = [col for col in df.columns if df[col].dtype == 'object']
    numerical = [col for col in df.columns if col not in categorical]

    for column_name in df.columns:
        if df[column_name].dtype == 'object':
            converted_column = pd.to_numeric(df[column_name], errors='coerce')
            if not converted_column.isna().all():
                df[column_name] = converted_column
                st.write(f"This column {column_name} has been converted to numirec to solve problem like '5' convert it to {5}")
                categorical.remove(column_name)
            else:
                label_encoder = LabelEncoder()
                encoded_values = label_encoder.fit_transform(df[column_name])
                df[column_name] = encoded_values
                label_encoders[column_name] = label_encoder
                st.write(f"The {column_name} column  has been labeled encoded")

    for column_name in categorical:
        selectbox_key = f"selectbox_categorical_{column_name}"
        categorical_handling = st.selectbox(f"Categorical Handling for {column_name}:", [None, 'Fill with Mode', 'Drop Nulls'], key=selectbox_key)
        st.write(f"The {column_name} column  has {df[column_name].isnull().sum()} Nulls")
        if categorical_handling == 'Fill with Mode':
            df[column_name].fillna(df[column_name].mode()[0], inplace=True)
            st.write('fill_mode')
            st.write(f"The {column_name} column didn't has any Nulls ({df[column_name].isnull().sum()}) after making {categorical_handling} Technique")
        elif categorical_handling == 'Drop Nulls':
            df.dropna(subset=[column_name], inplace=True)
            st.write('drop_nulls')
            st.write(f"The {column_name} column didn't has any Nulls ({df[column_name].isnull().sum()}) after making {categorical_handling} Technique")

    for column_name in numerical:
        selectbox_key = f"selectbox_numerical_{column_name}"
        numerical_handling = st.selectbox(f"Numerical Handling for {column_name}:", [None, 'Fill with Mean', 'Fill with Median', 'Drop Nulls'], key=selectbox_key)
        st.write(f"The {column_name} column  has {df[column_name].isnull().sum()} Nulls")
        if numerical_handling == 'Fill with Mean':
            df[column_name].fillna(df[column_name].mean(), inplace=True)
            st.write('fill_mean')
            st.write(f"The {column_name} column didn't has any Nulls ({df[column_name].isnull().sum()}) after making {numerical_handling} Technique")
        elif numerical_handling == 'Fill with Median':
            df[column_name].fillna(df[column_name].median(), inplace=True)
            st.write('fill_median')
            st.write(f"The {column_name} column didn't has any Nulls ({df[column_name].isnull().sum()}) after making {numerical_handling} Technique")
        elif numerical_handling == 'Drop Nulls':
            df.dropna(subset=[column_name], inplace=True)
            st.write('drop_nulls')
            st.write(f"The {column_name} column didn't has any Nulls ({df[column_name].isnull().sum()}) after making {numerical_handling} Technique")

    return df, label_encoders
