# Classification ML Project: Diabetes Prediction

Welcome to the Classification ML Project repository! In this project, we aim to predict diabetes based on various health indicators using machine learning models.

## Project Overview

This project follows the complete lifecycle of a machine learning project, including data preparation, model selection, training, testing, and evaluation. Here's a brief overview of the steps involved:

1. **Data Preparation**: We start by loading the dataset (`diabetes.csv`) into a Pandas DataFrame. We inspect the data, check for missing values, and split it into independent features (`x`) and the dependent variable (`y`).

2. **Model Selection**: We experiment with three different classification algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

3. **Model Training**: We train each model using the training data (`x_train` and `y_train`).

4. **Model Testing**: We evaluate the performance of each model using the testing data (`x_test` and `y_test`). We measure accuracy as our evaluation metric.

## Repository Structure

- **diabetes.csv**: Dataset containing health indicators and diabetes outcomes.
- **main.py**: Python code for the classification ML project.
- **README.md**: You are here! This document provides an overview of the project and instructions for usage.

## Usage

1. Clone this repository to your local machine using `git clone`.
2. Open `main.py` in Jupyter Notebook or any compatible environment.
3. Run the notebook cells to execute the code step by step.
4. Explore the code, experiment with different algorithms, and analyze the results.

## Dependencies

- Python 3.x
- Pandas
- scikit-learn
