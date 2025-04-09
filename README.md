# Air Quality Index (AQI) Forecasting Project

**Authors:**
* Andres Ferreira
* Ivan-codes99

## Overview

This project aims to develop and evaluate machine learning models for accurately forecasting the Air Quality Index (AQI). By leveraging historical air quality data and applying various regression techniques, we sought to build a reliable system for predicting future AQI levels.

## Key Steps and Findings

1.  **Data Loading and Exploration:**
    * The project began with loading and exploring an air quality dataset.
    * **Dataset Source:** [US Pollution Dataset on Kaggle](https://www.kaggle.com/datasets/sogun3/uspollution?resource=download)
    * Initial analysis focused on understanding the data's structure, features, and potential patterns.

2.  **Data Preprocessing:**
    * Missing values (NaN) in Sulfur Dioxide (SO2) and Carbon Monoxide (CO) were imputed using linear regression.
    * Zero and negative values were removed from the dataset.
    * Outliers were handled using the Interquartile Range (IQR) method, with capping applied to values exceeding the IQR boundaries, while allowing extreme values up to 200 to preserve potentially significant readings.

3.  **Baseline Model Development:** Four baseline regression models were implemented and evaluated:
    * Linear Regression
    * Random Forest Regression
    * Ridge Regression
    * XGBRegressor (with initial parameters)

4.  **Baseline Performance Evaluation:** The performance of each baseline model was assessed using key regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).

5.  **Hyperparameter Tuning (XGBoost):** A Grid Search approach with 5-fold cross-validation was employed to optimize the hyperparameters of the XGBoost Regressor, which showed promising initial performance. The following parameter grid was explored:
    ```python
    param_grid_xgb = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    ```
    The best parameters found were:
    ```
    Best parameters for XGBoost Regressor: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 1.0}
    ```
    The best cross-validation score (Negative MSE) was: `-2.2736`.

6.  **Tuned XGBoost Performance:** The XGBoost model trained with the optimized hyperparameters demonstrated significant improvements compared to the baseline, achieving the following performance on the test set:
    ```
    Mean Squared Error (MSE) of Best XGBoost Regressor: 2.3270
    Mean Absolute Error (MAE) of Best XGBoost Regressor: 0.1936
    Root Mean Squared Error (RMSE) of Best XGBoost Regressor: 1.5254
    R^2 Score (R^2) of Best XGBoost Regressor: 0.9942
    ```

## Challenges & Lessons Learned

* **Data Preprocessing:** Handling missing values (specifically for SO2 and CO), zero values, and outliers (with a threshold of 200 for extreme values) required careful consideration to avoid data loss or distortion. Linear regression was used for SO2 and CO imputation.
* **Computational Limitations:** Expanding the Grid Search for XGBoost and performing it for all models was limited by available computational resources. We prioritized tuning the best-performing baseline model (XGBoost).
* **Key Lessons:** Thorough data exploration is crucial. Outlier capping can be a balanced approach. Strategic hyperparameter tuning is necessary with limited resources. The machine learning process is iterative. Clear documentation is essential.

## Future Work

* Explore advanced time series models (ARIMA, LSTMs).
* Enhance features (lagged variables, rolling statistics, interaction terms, external data).
* Investigate ensemble methods.
* Experiment with more efficient hyperparameter tuning techniques (Randomized Search, Bayesian Optimization).
* Consider scalability for larger datasets and real-time predictions.
* Explore causality analysis and spatial modeling if applicable.
* Focus on predicting AQI threshold exceedances.
* Develop a user-friendly interface for accessing forecasts.

## Getting Started (Optional - If someone else wants to run your code)

1.  **Prerequisites:** Ensure you have Python 3 and the necessary libraries installed (e.g., pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn). You can install them using pip:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```
2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
    ```
3.  **Navigate to the notebooks directory:**
    ```bash
    cd notebooks
    jupyter notebook
    ```
