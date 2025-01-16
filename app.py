import streamlit as st
import automatic_data_preprocessing
import data_visualization
import manual_classification_algorithm
import manual_regression_algorithm
import manual_data_preprocessing
import pandas as pd
import automatic_classification
import automatic_regression

def main():
    st.title("Data Preprocessing and Machine Learning")

    file_path = st.file_uploader("Select a file (CSV or Excel):", type=["csv", "xlsx"])
    if file_path:
        st.write("Selected file:", file_path.name)
        
        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path)
            st.write(f"The data is in .csv")
        elif file_path.name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            st.write(f"The data is in .xlsx")
        else:
            st.error("Unsupported file format. Please select a CSV or Excel file.")
            return None, None
        
        preprocessing_method = st.selectbox("Choose preprocessing method", ["Select method", "Automatic", "Manual"])

        if preprocessing_method == "Select method":
            processed_df = None
            label_encoders = None  
            st.warning("Please choose a preprocessing method.")

        elif preprocessing_method == "Automatic":
            processed_df, label_encoders = automatic_data_preprocessing.preprocess_data(df)

        elif preprocessing_method == "Manual":
            processed_df, label_encoders = manual_data_preprocessing.preprocess_data(df)

        if processed_df is not None:
            st.write("Processed Data Types:")
            st.write(processed_df.dtypes)

            st.write("Label Encoders:")
            st.write(label_encoders)

            columns_to_drop = st.multiselect("Select columns to drop:", processed_df.columns)
            if columns_to_drop:
                processed_df.drop(columns=columns_to_drop, axis=1, inplace=True)
                label_encoders = {key: value for key, value in label_encoders.items() if key not in columns_to_drop}
                st.write("Columns dropped successfully:", columns_to_drop)
            else:
                st.write("No columns were dropped.")

            target_column = st.selectbox("Select your target column:", processed_df.columns)

            if st.checkbox("Make some of Data visualization"):
                data_visualization.visualization(processed_df, label_encoders, target_column)

            modelling_method = st.selectbox("Choose modelling method", ["Select method", "Automatic", "Manual"])

            if modelling_method == "Select method":
                st.warning("Please choose a preprocessing method.")
                
            elif modelling_method == "Automatic":
                if target_column in label_encoders:
                    automatic_classification.auto_sklearn(processed_df, target_column)
                else:
                    automatic_regression.auto_sklearn(processed_df, target_column)

            elif modelling_method == "Manual":
                problem_type = st.radio("Choose the problem type:", ("Classification", "Regression"))
                if problem_type == "Classification":
                    manual_classification_algorithm.run_classification_machine_learning(processed_df, target_column)
                else:
                    manual_regression_algorithm.run_regression_machine_learning(processed_df, target_column)

if __name__ == "__main__":
    main()
