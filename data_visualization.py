import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def visualization(data, label_encoders, target_column):

    sns.set(style="whitegrid")

    inverse_label_encoders = list(label_encoders.keys())
    for i in range(len(inverse_label_encoders)):
        data[inverse_label_encoders[i]] = label_encoders[inverse_label_encoders[i]].inverse_transform(data[inverse_label_encoders[i]])

    st.write(data.head())

    all_columns = data.columns.tolist()
    all_columns.remove(target_column)

    for column in all_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=data, x=column, bins=20, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        st.pyplot(plt)

    def Visualization(data):
        # Display summary statistics
        st.subheader("Summary Statistics:")
        st.write(data.describe())

        # Display data visualization
        st.subheader("Data Visualization:")

        # 1. Histogram for numerical columns
        numerical_cols = data.select_dtypes(include='number').columns
        for col in numerical_cols:
            st.write(f"### {col} Histogram")
            plt.figure(figsize=(8, 6))
            sns.histplot(data[col], kde=True)
            st.pyplot()
        
        # 2. Box plots for numerical columns
        st.write("### Box Plots for Numerical Columns")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data[numerical_cols])
        plt.xticks(rotation=45)
        st.pyplot()

        # 3. Pair plot for numerical columns (scatter plots and histograms)
        st.write("### Pair Plot for Numerical Columns")
        numerical_subset = data[numerical_cols]
        sns.set(style="ticks")
        sns.pairplot(numerical_subset, diag_kind='kde')
        st.pyplot()

        # 4. Count plot for categorical columns
        categorical_cols = data.select_dtypes(exclude='number').columns
        for col in categorical_cols:
            st.write(f"### {col} Count Plot")
            plt.figure(figsize=(8, 6))
            sns.countplot(data=data, x=col)
            plt.xticks(rotation=45)
            st.pyplot()

        # 5. Correlation heatmap for numerical columns
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        corr_matrix = numerical_subset.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot()
    Visualization(data)
    inverse_label_encoders = list(label_encoders.keys())
    for i in range(len(inverse_label_encoders)):
        data[inverse_label_encoders[i]] = label_encoders[inverse_label_encoders[i]].fit_transform(data[inverse_label_encoders[i]])
