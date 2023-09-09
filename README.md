# Streamlit-ML-Toolkit

Streamlit-ML-Toolkit is a Python project that aims to simplify the process of building and evaluating machine learning pipelines. The project provides an automated pipeline that performs data preprocessing, feature selection, algorithm selection, and model evaluation, with the flexibility to choose between classification and regression tasks.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Video Overview](#video-overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [Project Coming Updates](#project-coming-updates)

## Introduction

Welcome to the Streamlit-ML-Toolkit! This Python project is designed to simplify the complex process of building and evaluating machine learning pipelines. Whether you're a data scientist, a machine learning enthusiast, or someone looking to streamline your data analysis tasks, this toolkit is here to assist you. It provides an automated pipeline that takes care of data preprocessing, feature selection, algorithm selection, and model evaluation, all while offering the flexibility to choose between classification and regression tasks.

## Features

- Automated data preprocessing including handling missing values, label encoding, and data type conversion.
- Interactive GUI for selecting and dropping columns, and choosing the target column.
- Visualization of data distribution using histograms.
- Customize the visualization by selecting different numerical columns to display histograms
- Option to choose between classification and regression tasks.
- Selection of various machine learning algorithms for the chosen task.
- Model evaluation metrics displayed through a popup window.

## Video Overview

🎥 Watch this short video to see Streamlit-ML-Toolkit in action:

[![AutoML-Pipeline Video](https://img.youtube.com/vi/EVEbTLVU-Ws/0.jpg)](https://www.youtube.com/watch?v=EVEbTLVU-Ws)

[Click here to watch the video](https://www.youtube.com/watch?v=EVEbTLVU-Ws)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Mostafa-Tarek-00/Streamlit-ML-Toolkit.git
   ```

2. Install the required libraries by running:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn PySimpleGUI xgboost streamlit
   ```

3. Run the `app.py` script to launch the Streamlit-ML-Toolkit:
   ```bash
   streamlit run app.py
   ```
   
4. Follow the interactive GUI prompts to preprocess data, select columns, choose the target column, and run the selected algorithm.

## Usage

1. Launch the app by running the `app.py` script.
2. Select a dataset file in CSV or Excel format.
3. Preprocess the data by handling missing values and label encoding.
4. Choose columns to drop if necessary.
5. Select the target column for your task.
6. Choose between a classification or regression task.
7. Select an algorithm from the available options.
8. After running the algorithm, the app will display evaluation metrics.

## Contributing

welcome contributions to improve this project. To contribute, please follow these guidelines:

1. **Fork the Repository:** Click the "Fork" button at the top right of this repository's page.

2. **Create a New Branch:** Create a new branch for your feature or bug fix with a descriptive name.

3. **Make Changes:** Make your changes in your forked repository. Please ensure that your code adheres to the project's coding standards.

4. **Commit Changes:** Commit your changes with clear and descriptive commit messages.

5. **Push Changes:** Push your changes to your forked repository.

6. **Submit a Pull Request:** Submit a pull request to the main repository. In your pull request, describe the changes you've made and provide any relevant context.

7. **Review and Feedback:** The project maintainers will review your pull request, provide feedback, and work with you to address any issues.

## Project Coming Updates

- **First**: Introduced user-configurable parameters, including handling unbalanced data columns, improved data preprocessing, and enhanced visualization options.
- **Second**: Introduced user-configurable parameters, including handling unbalanced data columns, improved data preprocessing, and enhanced visualization options. Users can now adjust model parameters and combine multiple models for better results.

## Note

- The project is intended for educational purposes and might require further refinement for production use.
- The available algorithms and evaluation metrics can be extended based on your needs.

