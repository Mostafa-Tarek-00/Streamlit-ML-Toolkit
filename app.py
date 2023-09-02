import streamlit as st
import data_preprocessing
import data_visualization
import classification
import regression

def main():
    st.title("Data Preprocessing and Machine Learning")

    file_path = st.file_uploader("Select a file (CSV or Excel):", type=["csv", "xlsx"])
    if file_path:
        st.write("Selected file:", file_path.name)

        processed_df, label_encoders = data_preprocessing.preprocess_data(file_path)

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
                
            if st.radio("Choose the problem type:", ("Classification", "Regression")) == "Classification":
                classification.run_classification_machine_learning(processed_df, label_encoders, target_column)
            else:
                regression.run_regression_machine_learning(processed_df, label_encoders, target_column)

if __name__ == "__main__":
    main()
