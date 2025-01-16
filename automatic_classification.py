import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

def auto_sklearn(processed_df, target_column):

    # Set up the Streamlit app title and header
    st.title('Model Comparison with scikit-learn')
    st.header('Comparison of Classification Models')

    # Split the data into features and target
    X = processed_df.drop(target_column, axis=1)
    y = processed_df[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # List of classifiers to compare
    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'Dummy Classifier': DummyClassifier(strategy='most_frequent')
    }

    # Initialize a dictionary to store accuracy results
    model_results = []

    # Train and evaluate each model
    for model_name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_results.append({'Model': model_name, 'Accuracy': accuracy})

    # Convert the results into a pandas dataframe
    results_df = pd.DataFrame(model_results)

    # Sort the results by accuracy
    results_df = results_df.sort_values(by='Accuracy', ascending=False)

    # Display the comparison table in Streamlit
    st.subheader('Comparison of Classification Models')
    st.table(results_df)

    # Display the best model
    best_model = results_df.iloc[0]
    st.subheader(f"Best Model Based on Accuracy is {best_model['Model']} with {best_model['Accuracy']:.4f}")
