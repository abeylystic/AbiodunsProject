
import os
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
import copy
from pgmpy.estimators import PC
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import numpy as np
from linearmodels.panel import PanelOLS
from itertools import combinations


# Function to calculate AIC
def calculate_aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

# Function to calculate BIC
def calculate_bic(n, rss, k):
    return n * np.log(rss / n) + k * np.log(n)

# Function to calculate HQIC
def calculate_hqic(n, rss, k):
    return n * np.log(rss / n) + 2 * k * np.log(np.log(n))

# Function for forward stepwise selection
def forward_stepwise_selection(df, dependent_var, fixed_predictors, potential_predictors):
    initial_formula = f'{dependent_var} ~ ' + ' + '.join(fixed_predictors)
    best_model = initial_formula
    best_aic = float('inf')

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Formula', 'AIC'])

    # Fit the initial model
    model = PanelOLS.from_formula(initial_formula, df, drop_absorbed=True, check_rank=False)
    results = model.fit()
    n = results.nobs
    rss = np.sum(results.resids ** 2)
    num_params = results.params.shape[0]
    best_aic = calculate_aic(n, rss, num_params)
    results_df = results_df.append({'Formula': initial_formula, 'AIC': best_aic}, ignore_index=True)

    # Forward stepwise selection
    for k in range(1, len(potential_predictors) + 1):
        for subset in combinations(potential_predictors, k):
            formula = f'{dependent_var} ~ ' + ' + '.join(fixed_predictors + list(subset))
            model = PanelOLS.from_formula(formula, df, drop_absorbed=True, check_rank=False)
            results = model.fit()
            n = results.nobs
            rss = np.sum(results.resids ** 2)
            num_params = results.params.shape[0]
            aic = calculate_aic(n, rss, num_params)
            results_df = results_df.append({'Formula': formula, 'AIC': aic}, ignore_index=True)
            if aic < best_aic:
                best_aic = aic
                best_model = formula

    results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return best_model, best_aic, results_df


# Function to perform backward stepwise selection
def backward_stepwise_selection(df, dependent_var, fixed_predictors, potential_predictors):
    initial_predictors = fixed_predictors + potential_predictors
    initial_formula = f'{dependent_var} ~ ' + ' + '.join(initial_predictors)
    best_model = initial_formula
    best_aic = float('inf')

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Formula', 'AIC'])

    # Fit the initial model
    model = PanelOLS.from_formula(initial_formula, df, drop_absorbed=True, check_rank=False)
    results = model.fit()
    n = results.nobs
    rss = np.sum(results.resids ** 2)
    num_params = results.params.shape[0]
    best_aic = calculate_aic(n, rss, num_params)
    results_df = results_df.append({'Formula': initial_formula, 'AIC': best_aic}, ignore_index=True)

    current_predictors = initial_predictors

    # Backward stepwise selection
    improved = True
    while improved and len(current_predictors) > len(fixed_predictors):
        improved = False
        for predictor in current_predictors:
            if predictor not in fixed_predictors:
                candidate_predictors = [p for p in current_predictors if p != predictor]
                candidate_formula = f'{dependent_var} ~ ' + ' + '.join(candidate_predictors)
                model = PanelOLS.from_formula(candidate_formula, df, drop_absorbed=True, check_rank=False)
                results = model.fit()
                n = results.nobs
                rss = np.sum(results.resids ** 2)
                num_params = results.params.shape[0]
                aic = calculate_aic(n, rss, num_params)
                results_df = results_df.append({'Formula': candidate_formula, 'AIC': aic}, ignore_index=True)
                if aic < best_aic:
                    best_aic = aic
                    best_model = candidate_formula
                    current_predictors = candidate_predictors
                    improved = True

    results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return best_model, best_aic, results_df


# def get_model_summary(formula, df):
#     model = PanelOLS.from_formula(formula, df, drop_absorbed=True, check_rank=False)
#     results = model.fit()
#     return results


# Function to plot residuals vs predicted values
def plot_residuals_vs_predicted(results, df):
    predicted_values = results.fitted_values
    residuals = results.resids
    
    # Plot the histogram of the residuals
    fig, ax = plt.subplots(figsize=(20, 10))
    residuals.plot.hist(bins=100, ax=ax)
    plt.title("Residuals Histogram")
    plt.show()
#     plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.scatter(predicted_values, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    plt.close()



# Function to plot residuals vs independent variables
def plot_residuals_vs_independent_vars(results, df, fixed_predictors, potential_predictors):
    residuals = results.resids
    residuals = residuals[~residuals.index.duplicated(keep='first')]  # Remove duplicates from the index
    df['Residuals'] = residuals

    for predictor in fixed_predictors + potential_predictors:
        if predictor in df.columns:
            aligned_residuals = residuals.reindex(df.index)
            corr = df.corr().round(3)[predictor]["Residuals"]
            plt.figure(figsize=(12, 6))
            plt.scatter(df[predictor], aligned_residuals)
            plt.xlabel(predictor)
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs {predictor}, Corr: {corr}')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.show()


# Function to plot residuals vs dependent variable
def plot_residuals_vs_dependent_var(results, df, dependent_var='unem', title="Residuals vs Dependent Variable"):
    residuals = results.resids
    residuals = residuals[~residuals.index.duplicated(keep='first')]  # Remove duplicates from the index
    df['Residuals'] = residuals

    aligned_residuals = residuals.reindex(df.index)
    corr = df.corr().round(3)[dependent_var]["Residuals"]
    plt.figure(figsize=(12, 6))
    plt.scatter(df[dependent_var], aligned_residuals)
    plt.xlabel(f'Dependent Variable ({dependent_var})')
    plt.ylabel('Residuals')
    plt.title(f'{title}({dependent_var}), Corr: {corr}')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    
    
# Function to get model summary using PanelOLS
def get_model_summary(formula, df, model_type):
    if model_type == "PanelOLS":
        model = PanelOLS.from_formula(formula, df, drop_absorbed=True, check_rank=False)
    elif model_type == "PooledOLS":
        model = PooledOLS.from_formula(formula, df, check_rank=False)
    results = model.fit()
    return results


# Function to calculate average squared correlations
def calculate_avg_squared_correlations(results, df, independent_vars):
    residuals = results.resids
    squared_correlations = []
    for var in independent_vars:
        correlation = np.corrcoef(residuals, df[var])[0, 1]
        squared_correlation = correlation ** 2
        squared_correlations.append(squared_correlation)
    avg_squared_correlation = np.mean(squared_correlations)
    return avg_squared_correlation, squared_correlations
