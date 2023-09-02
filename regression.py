import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def run_regression_machine_learning(processed_df, label_encoders, target_column):
    X = processed_df.drop(target_column, axis=1)
    y = processed_df[target_column]

    algorithms = ["Linear Regression", "Lasso", "Ridge", "XGBoost", "K-Nearest Neighbors", "Support Vector Machine"]
    chosen_algorithm = st.selectbox("Select an algorithm:", algorithms)

    if st.button("Run Regression"):
        if chosen_algorithm == "Linear Regression":
            model = LinearRegression()
        elif chosen_algorithm == "Lasso":
            model = Lasso()
        elif chosen_algorithm == "Ridge":
            model = Ridge()
        elif chosen_algorithm == "XGBoost":
            model = XGBRegressor()
        elif chosen_algorithm == "K-Nearest Neighbors":
            model = KNeighborsRegressor()
        elif chosen_algorithm == "Support Vector Machine":
            model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        else:
            st.error("Invalid algorithm choice. Please select a valid algorithm.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.write(f"R2 Score: {r2:.2f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        st.write(f"Mean Squared Error: {mse:.2f}")
