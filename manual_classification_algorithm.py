import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def run_classification_machine_learning(processed_df, target_column):
    X = processed_df.drop(target_column, axis=1)
    y = processed_df[target_column]

    algorithms = ["Logistic Regression", "K-Nearest Neighbors", "XGBoost", "Support Vector Machine"]
    chosen_algorithm = st.selectbox("Select an algorithm:", algorithms)

    if st.button("Run Classification"):
        if chosen_algorithm == "Logistic Regression":
            model = LogisticRegression()
        elif chosen_algorithm == "XGBoost":
            model = XGBClassifier()
        elif chosen_algorithm == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        elif chosen_algorithm == "Support Vector Machine":
            model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        else:
            st.error("Invalid algorithm choice. Please select a valid algorithm.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1score = f1_score(y_test, y_pred, average="macro")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1score:.2f}")
