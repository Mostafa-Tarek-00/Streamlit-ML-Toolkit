import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def auto_sklearn(processed_df, target_column):
    # Set up the Streamlit app title and header
    st.title('Model Comparison with Scikit-learn')
    st.header('Comparison of Regression Models')

    # Split the data into training and test sets
    X = processed_df.drop(target_column, axis=1)
    y = processed_df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a list of models to compare
    models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'Support Vector Regressor': SVR(),
        'Decision Tree': DecisionTreeRegressor()
    }

    results = []

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': model_name,
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        })

    # Convert results to a DataFrame for display
    results_df = pd.DataFrame(results)

    # Display the table of compared models
    st.subheader('Comparison of Regression Models')
    st.table(results_df)

    # Display the best model based on R2
    best_model = results_df.loc[results_df['R2'].idxmax()]
    st.subheader('Best Model Based on R2')
    st.write(f"The best model is {best_model['Model']} with an R2 score of {best_model['R2']:.4f}")
