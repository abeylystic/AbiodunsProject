
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
import pingouin as pg
import statsmodels.api as sm
import geopandas
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from linearmodels.panel import PooledOLS    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import plotly.io as pio


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


import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout  # Using pygraphviz layout
# Function to buil skeleton
# def build_skeleton(df, undirected_graph):
#     # Function to build the skeleton (undirected graph) of the dataframe
#     # This is a placeholder implementation
#     # Replace it with the actual logic for building the skeleton
#     for i, var1 in enumerate(df.columns):
#         for var2 in df.columns[i + 1:]:
#             # Example criterion for adding an edge (replace with actual criterion)
#             if abs(df[var1].corr(df[var2])) > 0.5:
#                 undirected_graph[var1].append(var2)
#                 undirected_graph[var2].append(var1)
#     return undirected_graph

# Function to create undirected graph
# def graph_undirected_DAG(undirected_graph, title, ax):
#     graph = nx.Graph()
#     for node, edges in undirected_graph.items():
#         for edge in edges:
#             graph.add_edge(node, edge)
#     pos = graphviz_layout(graph, prog='neato')
#     nx.draw(graph, pos, ax=ax, node_color="C0", node_size=1500, with_labels=True, arrows=False, font_size=10, alpha=1, font_color="white")
#     ax.set_title(title, fontsize=12)

# # Function to convert undirected graph to directed
# def convert_to_directed(undirected_graph):
#     # Function to convert the undirected graph to a directed acyclic graph (DAG)
#     # This is a placeholder implementation
#     # Replace it with the actual logic for converting to a DAG
#     dag = nx.DiGraph()
#     for node, edges in undirected_graph.items():
#         for edge in edges:
#             dag.add_edge(node, edge)
#     return dag

# # Function to plot converted dag for specific dataframe
# def plot_dag_for_dataframe(df, exclude_columns=None, ax=None, title=None):
#     if exclude_columns is not None:
#         # Check if columns exist before dropping
#         drop_columns = [col for col in exclude_columns if col in df.columns]
#         df = df.drop(columns=drop_columns)

#     # Build undirected graph (skeleton)
#     undirected_graph = {key: [] for key in df.columns}
#     undirected_graph = build_skeleton(df, undirected_graph)
    
#     # Plot undirected graph
# #     graph_undirected_DAG(undirected_graph, title=f"Undirected Graph: {title}", ax=ax)
    
#     # Convert undirected graph to directed acyclic graph (DAG)
#     dag = convert_to_directed(undirected_graph)
    
#     # Plot DAG
#     pos = graphviz_layout(dag, prog='dot')
#     nx.draw(dag, pos, ax=ax, node_color="C0", node_size=1500, with_labels=True, arrows=True, font_size=10, alpha=1, font_color="black")
#     ax.set_title(f"DAG: {title}", fontsize=12)
#     ax.axis("off")

# # Function to plot converted graph in grids
# def plot_dags_in_grid(dag_dict, exclude_columns=None):
#     num_plots = len(dag_dict)
#     num_cols = 2  # Number of columns in the grid layout
#     num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows

#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
#     axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

#     for i, (dag_name, dag_df) in enumerate(dag_dict.items()):
#         ax = axs[i]
#         plot_dag_for_dataframe(dag_df, exclude_columns=exclude_columns, ax=ax, title=dag_name)
    
#     # Hide any remaining empty subplots
#     for j in range(i + 1, num_cols * num_rows):
#         axs[j].axis('off')

#     plt.tight_layout()
#     plt.show()    

# Function to get residuals 
residuals = {}
def get_residuals(df, weights):
    for y_var in df.columns:
        X_vars = list(df.columns)
        X_vars.remove(y_var)
        X = df[X_vars].copy()
        # Initial estimate should include constant
        X["Constant"] = 0
        y = df[[y_var]]
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
        residuals["$\\epsilon_{" + y_var + "}$"] = results.resid
    return pd.DataFrame(residuals)


# Function to import geographical data
def import_geo_data(filename, index_col = "Date", FIPS_name = "FIPS"):
    # import county level shapefile
    map_data = geopandas.read_file(filename = filename,                                   
                                   index_col = index_col)
    # rename fips code to match variable name in COVID-19 data
    map_data.rename(columns={"State":"state"},
                    inplace = True)
    # Combine statefips and county fips to create a single fips value
    # that identifies each particular county without referencing the 
    # state separately
    map_data[FIPS_name] = map_data["STATEFP"].astype(str) + \
        map_data["COUNTYFP"].astype(str)
    map_data[FIPS_name] = map_data[FIPS_name].astype(np.int64)
    # set FIPS as index
    map_data.set_index(FIPS_name, inplace=True)
    
    return map_data


# Function to Analyse and compare wls and pooled regressions from a dictionary of dataframes
def analyze_wls_pooled_models(data_cluster_dict, dependent_var, k=5, shuffle=True, random_state=None, check_rank=False, num_iterations=10):
    if random_state is not None:
        np.random.seed(random_state)
    
    mse_results = []
    model_attributes = []

    for key, df in data_cluster_dict.items():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        y = df[dependent_var]
        X = df.drop(columns=[dependent_var, 'FIPS', 'weight', 'TimePeriod'])
        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'cluster' in col]) if not use_clusters else X

            # Ensure X_filtered contains only numeric data
            X_filtered = X_filtered.select_dtypes(include=[np.number])            
            
            if 'Nominal rates' in key:
                X_filtered = sm.add_constant(X_filtered)
                
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_filtered.columns
            vif_data["VIF"] = [variance_inflation_factor(X_filtered.dropna().values, i) for i in range(len(X_filtered.columns))]

            avg_beta_coefficients = []

            for _ in range(num_iterations):
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                mse_folds = []

                for train_index, test_index in kf.split(df):
                    X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

                    model = sm.WLS(y_train, X_train, weights=weights_train).fit()
                    y_pred = model.predict(X_test)
                    mse_folds.append(mean_squared_error(y_test, y_pred))
                    avg_beta_coefficients.append(model.params.values)

                avg_mse = np.mean(mse_folds)
                mse_results.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'Avg MSE': avg_mse, 'MSE Folds': mse_folds, 'R^2': model.rsquared})

            avg_beta_coefficients = np.mean(avg_beta_coefficients, axis=0)
            model_attributes.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'Beta Estimates': pd.Series(avg_beta_coefficients, index=model.params.index), 'R^2': model.rsquared})

            y_train_pooled = y_train.reset_index()
            y_train_pooled['TimePeriod'] = df.iloc[train_index]['TimePeriod'].values
            y_train_pooled.set_index(['index', 'TimePeriod'], inplace=True)
            y_train_pooled = y_train_pooled[dependent_var]

            y_test_pooled = y_test.reset_index()
            y_test_pooled['TimePeriod'] = df.iloc[test_index]['TimePeriod'].values
            y_test_pooled.set_index(['index', 'TimePeriod'], inplace=True)
            y_test_pooled = y_test_pooled[dependent_var]

            X_train_pooled = X_train.reset_index()
            X_train_pooled['TimePeriod'] = df.iloc[train_index]['TimePeriod'].values
            X_train_pooled.set_index(['index', 'TimePeriod'], inplace=True)

            X_test_pooled = X_test.reset_index()
            X_test_pooled['TimePeriod'] = df.iloc[test_index]['TimePeriod'].values
            X_test_pooled.set_index(['index', 'TimePeriod'], inplace=True)

            avg_pooled_beta_coefficients = []

            for _ in range(num_iterations):
                pooled_model = PooledOLS(y_train_pooled, X_train_pooled, check_rank=check_rank).fit()
                y_pred_pooled = pooled_model.predict(X_test_pooled)
                mse_pooled = mean_squared_error(y_test_pooled, y_pred_pooled)
                mse_results.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'PooledOLS', 'Avg MSE': mse_pooled, 'MSE Folds': mse_folds, 'R^2': pooled_model.rsquared})
                avg_pooled_beta_coefficients.append(pooled_model.params.values)

            avg_pooled_beta_coefficients = np.mean(avg_pooled_beta_coefficients, axis=0)
            model_attributes.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'PooledOLS', 'Beta Estimates': pd.Series(avg_pooled_beta_coefficients, index=pooled_model.params.index), 'R^2': pooled_model.rsquared})

    # Accessing results as a list
    results_list = []
    for attributes in model_attributes:
        results_list.append({
            'Dataset': attributes['Dataset'],
            'Clusters': attributes['Clusters'],
            'Model': attributes['Model'],
            'Beta Estimates': attributes['Beta Estimates'].values.tolist(),
#             'Avg MSE': attributes['Avg MSE'],
            'R^2': attributes['R^2']
        })

    
    
    result_df = pd.DataFrame()
    for attributes in model_attributes:
        dataset_name = attributes['Dataset']
        clusters = attributes['Clusters']
        model_name = attributes['Model']
        beta_estimates = attributes['Beta Estimates']
        avg_mse = next(result['Avg MSE'] for result in mse_results if result['Dataset'] == dataset_name and result['Clusters'] == clusters and result['Model'] == model_name)
        mse_folds = next(result['MSE Folds'] for result in mse_results if result['Dataset'] == dataset_name and result['Clusters'] == clusters and result['Model'] == model_name)
        r_squared = attributes['R^2']
        mse_series = pd.Series([r_squared, avg_mse] + mse_folds, index=["$R^2$", "Avg MSE"] + [f"MSE Fold {i+1}" for i in range(len(mse_folds))])
        combined_series = pd.concat([beta_estimates, mse_series], axis=0)
        result_df = pd.concat([result_df, combined_series], axis=1)
        result_df.rename(columns={result_df.columns[-1]: f"{dataset_name} - {clusters} - {model_name}"}, inplace=True)

    return result_df


# Function to run wls and pooledOLS without k-fold
def wls_pooled_model_analysis(data_cluster_dict, dependent_var, random_state=None, check_rank=False):
    if random_state is not None:
        np.random.seed(random_state)

    mse_results = []
    model_attributes = []

    for key, df in data_cluster_dict.items():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        y = df[dependent_var]
        X = df.drop(columns=[dependent_var, 'FIPS', 'weight', 'TimePeriod'])
        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'cluster' in col]) if not use_clusters else X

            # Ensure X_filtered contains only numeric data
            X_filtered = X_filtered.select_dtypes(include=[np.number])            
            
            if 'Nominal rates' in key:
                X_filtered = sm.add_constant(X_filtered)
                
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_filtered.columns
            vif_data["VIF"] = [variance_inflation_factor(X_filtered.dropna().values, i) for i in range(len(X_filtered.columns))]

            model = sm.WLS(y, X_filtered, weights=weights).fit()
            model_attributes.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'Beta Estimates': model.params, 'R^2': model.rsquared})

            y_pooled = y.reset_index()
            y_pooled['TimePeriod'] = df['TimePeriod'].values
            y_pooled.set_index(['index', 'TimePeriod'], inplace=True)
            y_pooled = y_pooled[dependent_var]

            X_pooled = X_filtered.reset_index()
            X_pooled['TimePeriod'] = df['TimePeriod'].values
            X_pooled.set_index(['index', 'TimePeriod'], inplace=True)

            pooled_model = PooledOLS(y_pooled, X_pooled, check_rank=check_rank).fit()
            model_attributes.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'PooledOLS', 'Beta Estimates': pooled_model.params, 'R^2': pooled_model.rsquared})

            
    # Accessing results as a list
    results_list = []
    for attributes in model_attributes:
        results_list.append({
            'Dataset': attributes['Dataset'],
            'Clusters': attributes['Clusters'],
            'Model': attributes['Model'],
            'Beta Estimates': attributes['Beta Estimates'].values.tolist(),
            'R^2': attributes['R^2']
        })
        
            
    result_df = pd.DataFrame()
    for attributes in model_attributes:
        dataset_name = attributes['Dataset']
        clusters = attributes['Clusters']
        model_name = attributes['Model']
        beta_estimates = attributes['Beta Estimates']
        r_squared = attributes['R^2']
        combined_series = pd.concat([beta_estimates, pd.Series(r_squared, index=["$R^2$"])], axis=0)
        result_df = pd.concat([result_df, combined_series], axis=1)
        result_df.rename(columns={result_df.columns[-1]: f"{dataset_name} - {clusters} - {model_name}"}, inplace=True)

    return result_df


# Function to perform k-fold on wls and pooledOLS and report all mse's
def analyze_wls_pooled_models_all_mse(data_cluster_dict, dependent_var, k=5, shuffle=True, random_state=None, check_rank=False, num_iterations=10):
    if random_state is not None:
        np.random.seed(random_state)
    
    mse_results = []
    model_attributes = []

    for key, df in data_cluster_dict.items():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        y = df[dependent_var]
        X = df.drop(columns=[dependent_var, 'FIPS', 'weight', 'TimePeriod'])
        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'cluster' in col]) if not use_clusters else X

            # Ensure X_filtered contains only numeric data
            X_filtered = X_filtered.select_dtypes(include=[np.number])            
            
            if 'Nominal rates' in key:
                X_filtered = sm.add_constant(X_filtered)
                
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_filtered.columns
            vif_data["VIF"] = [variance_inflation_factor(X_filtered.dropna().values, i) for i in range(len(X_filtered.columns))]

            avg_beta_coefficients = []
            mse_folds = []

            for _ in range(num_iterations):
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

                for train_index, test_index in kf.split(df):
                    X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

                    # WLS model
                    model = sm.WLS(y_train, X_train, weights=weights_train).fit()
                    y_pred = model.predict(X_test)
                    mse_folds.append(mean_squared_error(y_test, y_pred))
                    avg_beta_coefficients.append(model.params.values)

                avg_mse = np.mean(mse_folds)
                mse_results.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'Avg MSE': avg_mse, 'MSE Folds': mse_folds, 'R^2': model.rsquared})

            avg_beta_coefficients = np.mean(avg_beta_coefficients, axis=0)
            model_attributes.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'Beta Estimates': pd.Series(avg_beta_coefficients, index=model.params.index), 'R^2': model.rsquared})

            avg_pooled_beta_coefficients = []
            mse_folds_pooled = []

            for _ in range(num_iterations):
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

                for train_index, test_index in kf.split(df):
                    y_train_pooled = y.iloc[train_index].reset_index()
                    y_train_pooled['TimePeriod'] = df.iloc[train_index]['TimePeriod'].values
                    y_train_pooled.set_index(['index', 'TimePeriod'], inplace=True)
                    y_train_pooled = y_train_pooled[dependent_var]

                    y_test_pooled = y.iloc[test_index].reset_index()
                    y_test_pooled['TimePeriod'] = df.iloc[test_index]['TimePeriod'].values
                    y_test_pooled.set_index(['index', 'TimePeriod'], inplace=True)
                    y_test_pooled = y_test_pooled[dependent_var]

                    X_train_pooled = X_filtered.iloc[train_index].reset_index()
                    X_train_pooled['TimePeriod'] = df.iloc[train_index]['TimePeriod'].values
                    X_train_pooled.set_index(['index', 'TimePeriod'], inplace=True)

                    X_test_pooled = X_filtered.iloc[test_index].reset_index()
                    X_test_pooled['TimePeriod'] = df.iloc[test_index]['TimePeriod'].values
                    X_test_pooled.set_index(['index', 'TimePeriod'], inplace=True)

                    pooled_model = PooledOLS(y_train_pooled, X_train_pooled, check_rank=check_rank).fit()
                    y_pred_pooled = pooled_model.predict(X_test_pooled)
                    mse_folds_pooled.append(mean_squared_error(y_test_pooled, y_pred_pooled))
                    avg_pooled_beta_coefficients.append(pooled_model.params.values)

                avg_mse_pooled = np.mean(mse_folds_pooled)
                mse_results.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'PooledOLS', 'Avg MSE': avg_mse_pooled, 'MSE Folds': mse_folds_pooled, 'R^2': pooled_model.rsquared})

            avg_pooled_beta_coefficients = np.mean(avg_pooled_beta_coefficients, axis=0)
            model_attributes.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'PooledOLS', 'Beta Estimates': pd.Series(avg_pooled_beta_coefficients, index=pooled_model.params.index), 'R^2': pooled_model.rsquared})

    # Accessing results as a list
    results_list = []
    for attributes in model_attributes:
        results_list.append({
            'Dataset': attributes['Dataset'],
            'Clusters': attributes['Clusters'],
            'Model': attributes['Model'],
            'Beta Estimates': attributes['Beta Estimates'].values.tolist(),
#             'Avg MSE': attributes['Avg MSE'],
            'R^2': attributes['R^2']
        })

    
    
    result_df = pd.DataFrame()
    for attributes in model_attributes:
        dataset_name = attributes['Dataset']
        clusters = attributes['Clusters']
        model_name = attributes['Model']
        beta_estimates = attributes['Beta Estimates']
        avg_mse = next(result['Avg MSE'] for result in mse_results if result['Dataset'] == dataset_name and result['Clusters'] == clusters and result['Model'] == model_name)
        mse_folds = next(result['MSE Folds'] for result in mse_results if result['Dataset'] == dataset_name and result['Clusters'] == clusters and result['Model'] == model_name)
        r_squared = attributes['R^2']
        mse_series = pd.Series([r_squared, avg_mse] + mse_folds, index=["$R^2$", "Avg MSE"] + [f"MSE Fold {i+1}" for i in range(len(mse_folds))])
        combined_series = pd.concat([beta_estimates, mse_series], axis=0)
        result_df = pd.concat([result_df, combined_series], axis=1)
        result_df.rename(columns={result_df.columns[-1]: f"{dataset_name} - {clusters} - {model_name}"}, inplace=True)

    return result_df

# Function to perform k-fold on wls and pooledOLS and report the least mse
def analyze_wls_pooled_models_least_mse(data_cluster_dict, dependent_var, k=5, shuffle=True, random_state=None, check_rank=False, num_iterations=10):
    if random_state is not None:
        np.random.seed(random_state)
    
    model_attributes = []

    for key, df in data_cluster_dict.items():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        y = df[dependent_var]
        X = df.drop(columns=[dependent_var, 'FIPS', 'weight', 'TimePeriod'])
        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'cluster' in col]) if not use_clusters else X

            # Ensure X_filtered contains only numeric data
            X_filtered = X_filtered.select_dtypes(include=[np.number])            
            
            if 'Nominal rates' in key:
                X_filtered = sm.add_constant(X_filtered)
                
            all_mse_folds = []
            all_beta_coefficients = []

            for _ in range(num_iterations):
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                for train_index, test_index in kf.split(df):
                    X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

                    # WLS model
                    model = sm.WLS(y_train, X_train, weights=weights_train).fit()
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    all_mse_folds.append(mse)
                    all_beta_coefficients.append(model.params.values)

            # Select the least MSE across all iterations and folds
            least_mse_index = np.argmin(all_mse_folds)
            least_mse = all_mse_folds[least_mse_index]
            least_beta_coefficients = all_beta_coefficients[least_mse_index]

            model_attributes.append({
                'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 
                'Beta Estimates': pd.Series(least_beta_coefficients, index=model.params.index),
                'Least MSE': least_mse,
                'R^2': model.rsquared
            })

            all_pooled_mse_folds = []
            all_pooled_beta_coefficients = []

            for _ in range(num_iterations):
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                for train_index, test_index in kf.split(df):
                    y_train_pooled = y.iloc[train_index].reset_index()
                    y_train_pooled['TimePeriod'] = df.iloc[train_index]['TimePeriod'].values
                    y_train_pooled.set_index(['index', 'TimePeriod'], inplace=True)
                    y_train_pooled = y_train_pooled[dependent_var]

                    y_test_pooled = y.iloc[test_index].reset_index()
                    y_test_pooled['TimePeriod'] = df.iloc[test_index]['TimePeriod'].values
                    y_test_pooled.set_index(['index', 'TimePeriod'], inplace=True)
                    y_test_pooled = y_test_pooled[dependent_var]

                    X_train_pooled = X_filtered.iloc[train_index].reset_index()
                    X_train_pooled['TimePeriod'] = df.iloc[train_index]['TimePeriod'].values
                    X_train_pooled.set_index(['index', 'TimePeriod'], inplace=True)

                    X_test_pooled = X_filtered.iloc[test_index].reset_index()
                    X_test_pooled['TimePeriod'] = df.iloc[test_index]['TimePeriod'].values
                    X_test_pooled.set_index(['index', 'TimePeriod'], inplace=True)

                    pooled_model = PooledOLS(y_train_pooled, X_train_pooled, check_rank=check_rank).fit()
                    y_pred_pooled = pooled_model.predict(X_test_pooled)
                    mse_pooled = mean_squared_error(y_test_pooled, y_pred_pooled)
                    all_pooled_mse_folds.append(mse_pooled)
                    all_pooled_beta_coefficients.append(pooled_model.params.values)

            # Select the least MSE across all iterations and folds for PooledOLS
            least_mse_index_pooled = np.argmin(all_pooled_mse_folds)
            least_mse_pooled = all_pooled_mse_folds[least_mse_index_pooled]
            least_beta_coefficients_pooled = all_pooled_beta_coefficients[least_mse_index_pooled]

            model_attributes.append({
                'Dataset': key, 'Clusters': use_clusters, 'Model': 'PooledOLS', 
                'Beta Estimates': pd.Series(least_beta_coefficients_pooled, index=pooled_model.params.index),
                'Least MSE': least_mse_pooled,
                'R^2': pooled_model.rsquared
            })
            
    # Accessing results as a list
    results_list = []
    for attributes in model_attributes:
        results_list.append({
            'Dataset': attributes['Dataset'],
            'Clusters': attributes['Clusters'],
            'Model': attributes['Model'],
            'Beta Estimates': attributes['Beta Estimates'].values.tolist(),
            'Least MSE': attributes['Least MSE'],
            'R^2': attributes['R^2']
        })
        

    result_df = pd.DataFrame()
    for attributes in model_attributes:
        dataset_name = attributes['Dataset']
        clusters = attributes['Clusters']
        model_name = attributes['Model']
        beta_estimates = attributes['Beta Estimates']
        least_mse = attributes['Least MSE']
        r_squared = attributes['R^2']
        mse_series = pd.Series([r_squared, least_mse], index=["$R^2$", "Least MSE"])
        combined_series = pd.concat([beta_estimates, mse_series], axis=0)
        result_df = pd.concat([result_df, combined_series], axis=1)
        result_df.rename(columns={result_df.columns[-1]: f"{dataset_name} - {clusters} - {model_name}"}, inplace=True)

    return result_df


# Function to perform k-fold on wls and OLS and report the least mse
def analyze_wls_ols_models_least_mse(data_cluster_dict, dependent_var, k=5, shuffle=True, random_state=None, check_rank=True, num_iterations=10):
    if random_state is not None:
        np.random.seed(random_state)
    
    model_attributes = []

    for key, df in data_cluster_dict.items():
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        y = df[dependent_var]
        X = df.drop(columns=[dependent_var, 'FIPS', 'weight', 'TimePeriod'])
        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'cluster' in col]) if not use_clusters else X

            # Ensure X_filtered contains only numeric data
            X_filtered = X_filtered.select_dtypes(include=[np.number])            
            
            if 'Nominal rates' in key:
                X_filtered = sm.add_constant(X_filtered)
                
            all_mse_folds = []
            all_beta_coefficients = []

            for _ in range(num_iterations):
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                for train_index, test_index in kf.split(df):
                    X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

                    # WLS model
                    model = sm.WLS(y_train, X_train, weights=weights_train).fit()
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    all_mse_folds.append(mse)
                    all_beta_coefficients.append(model.params.values)

            # Select the least MSE across all iterations and folds
            least_mse_index = np.argmin(all_mse_folds)
            least_mse = all_mse_folds[least_mse_index]
            least_beta_coefficients = all_beta_coefficients[least_mse_index]

            model_attributes.append({
                'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 
                'Beta Estimates': pd.Series(least_beta_coefficients, index=model.params.index),
                'Least MSE': least_mse,
                'R^2': model.rsquared,
                'Residuals': model.resid  # Add residuals to the model attributes
            })

            all_ols_mse_folds = []
            all_ols_beta_coefficients = []

            for _ in range(num_iterations):
                kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                for train_index, test_index in kf.split(df):
                    X_train_ols, X_test_ols = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                    y_train_ols, y_test_ols = y.iloc[train_index], y.iloc[test_index]

                    ols_model = sm.OLS(y_train_ols, X_train_ols).fit()
                    y_pred_ols = ols_model.predict(X_test_ols)
                    mse_ols = mean_squared_error(y_test_ols, y_pred_ols)
                    all_ols_mse_folds.append(mse_ols)
                    all_ols_beta_coefficients.append(ols_model.params.values)

            # Select the least MSE across all iterations and folds for OLS
            least_mse_index_ols = np.argmin(all_ols_mse_folds)
            least_mse_ols = all_ols_mse_folds[least_mse_index_ols]
            least_beta_coefficients_ols = all_ols_beta_coefficients[least_mse_index_ols]

            model_attributes.append({
                'Dataset': key, 'Clusters': use_clusters, 'Model': 'OLS', 
                'Beta Estimates': pd.Series(least_beta_coefficients_ols, index=ols_model.params.index),
                'Least MSE': least_mse_ols,
                'R^2': ols_model.rsquared,
                'Residuals': ols_model.resid  # Add residuals to the model attributes
            })
            
    return model_attributes

# Function to analyse olw and wls with varying k-folds
def analyze_wls_ols_models_with_varying_folds(data_cluster_dict, dependent_var, start_k=5, max_k=10, shuffle=True, random_state=None, num_iterations=10):
    if random_state is not None:
        np.random.seed(random_state)
    
    results = {}

    for key, df in data_cluster_dict.items():
        model_attributes = []

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        y = df[dependent_var]
        X = df.drop(columns=[dependent_var, 'FIPS', 'weight', 'TimePeriod'])
        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'cluster' in col]) if not use_clusters else X

            if 'Nominal rates' in key or 'Nominal log' in key:
                X_filtered = sm.add_constant(X_filtered)

            all_mse_folds = []
            all_beta_coefficients = []

            for k in range(start_k, max_k + 1):
                for _ in range(num_iterations):
                    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                    for train_index, test_index in kf.split(df):
                        X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

                        # WLS model
                        model = sm.WLS(y_train, X_train, weights=weights_train).fit()
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        all_mse_folds.append(mse)
                        all_beta_coefficients.append(model.params.values)

                # Select the least 3 MSEs across all iterations and folds
                top_3_indices = np.argsort(all_mse_folds)[:3]
                top_3_mse = [all_mse_folds[i] for i in top_3_indices]
                avg_top_3_mse = np.mean(top_3_mse)
                top_3_betas = [all_beta_coefficients[i] for i in top_3_indices]
                avg_top_3_betas = np.mean(top_3_betas, axis=0)

                model_attributes.append({
                    'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'K': k, 
                    'Avg Beta Estimates': pd.Series(avg_top_3_betas, index=model.params.index),
                    'Avg Top 3 MSE': avg_top_3_mse,
                    'R^2': model.rsquared,
                    'Residuals': model.resid
                })

                all_ols_mse_folds = []
                all_ols_beta_coefficients = []

                for _ in range(num_iterations):
                    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                    for train_index, test_index in kf.split(df):
                        X_train_ols, X_test_ols = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                        y_train_ols, y_test_ols = y.iloc[train_index], y.iloc[test_index]

                        ols_model = sm.OLS(y_train_ols, X_train_ols).fit()
                        y_pred_ols = ols_model.predict(X_test_ols)
                        mse_ols = mean_squared_error(y_test_ols, y_pred_ols)
                        all_ols_mse_folds.append(mse_ols)
                        all_ols_beta_coefficients.append(ols_model.params.values)

                # Select the least 3 MSEs across all iterations and folds for OLS
                top_3_indices_ols = np.argsort(all_ols_mse_folds)[:3]
                top_3_mse_ols = [all_ols_mse_folds[i] for i in top_3_indices_ols]
                avg_top_3_mse_ols = np.mean(top_3_mse_ols)
                top_3_betas_ols = [all_ols_beta_coefficients[i] for i in top_3_indices_ols]
                avg_top_3_betas_ols = np.mean(top_3_betas_ols, axis=0)

                model_attributes.append({
                    'Dataset': key, 'Clusters': use_clusters, 'Model': 'OLS', 'K': k, 
                    'Avg Beta Estimates': pd.Series(avg_top_3_betas_ols, index=ols_model.params.index),
                    'Avg Top 3 MSE': avg_top_3_mse_ols,
                    'R^2': ols_model.rsquared,
                    'Residuals': ols_model.resid
                })
        
        results[key] = model_attributes

    return results


# Function to plot wls and ols caomparisons (bar charts)
def plot_model_comparisons(df, title_suffix):
    # Initialize colors for 'WLS' and 'OLS' models
    color_wls = 'skyblue'
    color_ols = 'salmon'

    # Extract variables and K values from the dataframe
    variables = set()
    k_values = set()
    for entry in df:
        if 'Avg Beta Estimates' in entry:
            variables.update(entry['Avg Beta Estimates'].keys())
        k_values.add(entry['K'])
    
    variables = sorted(variables)
    k_values = sorted(k_values)
    
    # Initialize a figure with subplots
    fig, axes = plt.subplots(2 * (len(variables) + 2), 1, figsize=(12, 12 * (len(variables) + 2)), sharex=True)
    
    for idx, var in enumerate(variables):
        for cluster_status in [True, False]:
            for model_idx, model in enumerate(['WLS', 'OLS']):
                var_values = []
                for entry in df:
                    if entry['Clusters'] == cluster_status and entry['Model'] == model:
                        k = entry['K']
                        if var in entry['Avg Beta Estimates']:
                            var_values.append(entry['Avg Beta Estimates'][var])
                        else:
                            var_values.append(0)  # Handle missing variable case
                if model == 'WLS':
                    axes[2 * idx + cluster_status].bar(np.arange(len(k_values)) + model_idx * 0.4, var_values, color=color_wls, width=0.4, alpha=0.7, label=model)
                elif model == 'OLS':
                    axes[2 * idx + cluster_status].bar(np.arange(len(k_values)) + model_idx * 0.4, var_values, color=color_ols, width=0.4, alpha=0.7, label=model)
            axes[2 * idx + cluster_status].set_xlabel('K values')
            axes[2 * idx + cluster_status].set_ylabel(f'Values')
            axes[2 * idx + cluster_status].set_title(f'{var} ({title_suffix}, Clusters={cluster_status})')
            axes[2 * idx + cluster_status].set_xticks(np.arange(len(k_values)) + 0.2)
            axes[2 * idx + cluster_status].set_xticklabels([f'K={k}' for k in k_values])  # Set the x-axis labels to include K values
            axes[2 * idx + cluster_status].grid(axis='y')
            axes[2 * idx + cluster_status].legend()
    
    # Include R^2 and Avg Top 3 MSE in the plot
    for cluster_status in [True, False]:
        for model_idx, model in enumerate(['WLS', 'OLS']):
            r_squared_values = [entry['R^2'] for entry in df if entry['Clusters'] == cluster_status and entry['Model'] == model]
            avg_top3_mse_values = [entry['Avg Top 3 MSE'] for entry in df if entry['Clusters'] == cluster_status and entry['Model'] == model]
            if model == 'WLS':
                axes[2 * len(variables) + cluster_status].bar(np.arange(len(k_values)) + model_idx * 0.4, r_squared_values, color=color_wls, width=0.4, alpha=0.7, label=model)
                axes[2 * len(variables) + 2 + cluster_status].bar(np.arange(len(k_values)) + model_idx * 0.4, avg_top3_mse_values, color=color_wls, width=0.4, alpha=0.7, label=model)
            elif model == 'OLS':
                axes[2 * len(variables) + cluster_status].bar(np.arange(len(k_values)) + model_idx * 0.4, r_squared_values, color=color_ols, width=0.4, alpha=0.7, label=model)
                axes[2 * len(variables) + 2 + cluster_status].bar(np.arange(len(k_values)) + model_idx * 0.4, avg_top3_mse_values, color=color_ols, width=0.4, alpha=0.7, label=model)
            
            axes[2 * len(variables) + cluster_status].set_xlabel('K values')
            axes[2 * len(variables) + cluster_status].set_ylabel('$R^2$')
            axes[2 * len(variables) + cluster_status].set_title(f'$R^2$ ({title_suffix}, Clusters={cluster_status})')
            axes[2 * len(variables) + cluster_status].set_xticks(np.arange(len(k_values)) + 0.2)
            axes[2 * len(variables) + cluster_status].set_xticklabels([f'K={k}' for k in k_values])  # Set the x-axis labels to include K values
            axes[2 * len(variables) + cluster_status].grid(axis='y')
            axes[2 * len(variables) + cluster_status].legend()
            
            axes[2 * len(variables) + 2 + cluster_status].set_xlabel('K values')
            axes[2 * len(variables) + 2 + cluster_status].set_ylabel('Avg Top 3 MSE')
            axes[2 * len(variables) + 2 + cluster_status].set_title(f'Avg Top 3 MSE ({title_suffix}, Clusters={cluster_status})')
            axes[2 * len(variables) + 2 + cluster_status].set_xticks(np.arange(len(k_values)) + 0.2)
            axes[2 * len(variables) + 2 + cluster_status].set_xticklabels([f'K={k}' for k in k_values])  # Set the x-axis labels to include K values
            axes[2 * len(variables) + 2 + cluster_status].grid(axis='y')
            axes[2 * len(variables) + 2 + cluster_status].legend()
    
    plt.tight_layout()
    plt.show()
    


# Function to plot dynamic line plots 
def plot_model_comparisons_with_dropdowns(results_dict):
    variables = set()
    k_values = set()
    for entry in results_dict['Nominal rates with clusters']:
        if 'Avg Beta Estimates' in entry:
            variables.update(entry['Avg Beta Estimates'].keys())
        k_values.add(entry['K'])
    
    variables = sorted(variables)
    k_values = sorted(k_values)

    # Create subplots: 3 rows, 2 columns
    fig = make_subplots(rows=3, cols=2, subplot_titles=[
        'Beta Estimates (OLS)', 'Beta Estimates (WLS)',
        'Avg Top 3 MSE (OLS)', 'Avg Top 3 MSE (WLS)',
        '$R^2 (OLS)$', '$R^2 (WLS)$'
    ], vertical_spacing=0.1, horizontal_spacing=0.1)

    def add_traces_to_fig(df, title_suffix):
        for var in variables:
            for cluster_status in [True, False]:
                for model in ['WLS', 'OLS']:
                    var_values = []
                    for entry in df:
                        if entry['Clusters'] == cluster_status and entry['Model'] == model:
                            k = entry['K']
                            if var in entry['Avg Beta Estimates']:
                                var_values.append(entry['Avg Beta Estimates'][var])
                            else:
                                var_values.append(0)  # Handle missing variable case

                    line_style = dict(dash='dash') if not cluster_status else dict()  # Dotted lines for Clusters=False
                    cluster_text = "Clusters=True" if cluster_status else "Clusters=False"
                    row, col = (1, 1) if model == 'OLS' else (1, 2)
                    fig.add_trace(go.Scatter(x=k_values, y=var_values, mode='lines+markers', name=f'{var} ({model}, {cluster_text}, {title_suffix})', line=line_style, visible=False), row=row, col=col)

        for cluster_status in [True, False]:
            for model in ['WLS', 'OLS']:
                r_squared_values = [entry['R^2'] for entry in df if entry['Clusters'] == cluster_status and entry['Model'] == model]
                avg_top3_mse_values = [entry['Avg Top 3 MSE'] for entry in df if entry['Clusters'] == cluster_status and entry['Model'] == model]
                
                line_style = dict(dash='dash') if not cluster_status else dict()  # Dotted lines for Clusters=False
                cluster_text = "Clusters=True" if cluster_status else "Clusters=False"
                row_r2, col_r2 = (3, 1) if model == 'OLS' else (3, 2)
                row_mse, col_mse = (2, 1) if model == 'OLS' else (2, 2)
                
                fig.add_trace(go.Scatter(x=k_values, y=avg_top3_mse_values, mode='lines+markers', name=f'Avg Top 3 MSE ({model}, {cluster_text}, {title_suffix})', line=line_style, visible=False), row=row_mse, col=col_mse)
                fig.add_trace(go.Scatter(x=k_values, y=r_squared_values, mode='lines+markers', name=f'R^2 ({model}, {cluster_text}, {title_suffix})', line=line_style, visible=False), row=row_r2, col=col_r2)

    add_traces_to_fig(results_dict['Nominal rates with clusters'], 'Nominal rates')
    add_traces_to_fig(results_dict['Nominal diff rates with clusters'], 'Nominal diff rates')

    # Update layout with dropdown and legend button
    fig.update_layout(
        title='Model Comparisons',
        showlegend=False,
        height=900, width=1400,  # Increase plot size
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'Nominal rates',
                        'method': 'update',
                        'args': [
                            {'visible': [i < len(fig.data) // 2 for i in range(len(fig.data))]},
                            {'title': 'Model Comparisons for Nominal rates'}
                        ]
                    },
                    {
                        'label': 'Nominal diff rates',
                        'method': 'update',
                        'args': [
                            {'visible': [i >= len(fig.data) // 2 for i in range(len(fig.data))]},
                            {'title': 'Model Comparisons for Nominal diff rates'}
                        ]
                    },
                    {
                        'label': 'Both',
                        'method': 'update',
                        'args': [
                            {'visible': [True for _ in range(len(fig.data))]},
                            {'title': 'Model Comparisons for Nominal rates and Nominal diff rates'}
                        ]
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.17,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            },
            {
                'buttons': [
                    {
                        'label': 'Show Legend',
                        'method': 'relayout',
                        'args': [{'showlegend': True}]
                    },
                    {
                        'label': 'Hide Legend',
                        'method': 'relayout',
                        'args': [{'showlegend': False}]
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.3,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            }
        ]
    )

    # Initialize the plot with 'Nominal rates' visible
    for i in range(len(fig.data) // 2):
        fig.data[i].visible = True

    return fig

# For all dataframes
def wls_ols_with_varying_folds(data_cluster_dict, dependent_var, start_k=5, max_k=10, shuffle=True, random_state=None, num_iterations=10):
    if random_state is not None:
        np.random.seed(random_state)
    
    results = {}

    for key, df in data_cluster_dict.items():
        model_attributes = []

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        y = df[dependent_var]
        X = df.drop(columns=[dependent_var, 'FIPS', 'weight', 'TimePeriod'])
        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'cluster' in col]) if not use_clusters else X

            # Ensure X_filtered contains only numeric data
            X_filtered = X_filtered.select_dtypes(include=[np.number])            
            
            if 'Nominal rates' in key or 'Nominal log' in key:
                X_filtered = sm.add_constant(X_filtered)

            all_mse_folds = []
            all_beta_coefficients = []

            for k in range(start_k, max_k + 1):
                for _ in range(num_iterations):
                    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                    for train_index, test_index in kf.split(df):
                        X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

                        # WLS model
                        model = sm.WLS(y_train, X_train, weights=weights_train).fit()
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        all_mse_folds.append(mse)
                        all_beta_coefficients.append(model.params.values)

                # Select the least 3 MSEs across all iterations and folds
                top_3_indices = np.argsort(all_mse_folds)[:3]
                top_3_mse = [all_mse_folds[i] for i in top_3_indices]
                avg_top_3_mse = np.mean(top_3_mse)
                top_3_betas = [all_beta_coefficients[i] for i in top_3_indices]
                avg_top_3_betas = np.mean(top_3_betas, axis=0)

                model_attributes.append({
                    'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'K': k, 
                    'Avg Beta Estimates': pd.Series(avg_top_3_betas, index=model.params.index),
                    'Avg Top 3 MSE': avg_top_3_mse,
                    'R^2': model.rsquared,
                    'Residuals': model.resid
                })

                all_ols_mse_folds = []
                all_ols_beta_coefficients = []

                for _ in range(num_iterations):
                    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
                    for train_index, test_index in kf.split(df):
                        X_train_ols, X_test_ols = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                        y_train_ols, y_test_ols = y.iloc[train_index], y.iloc[test_index]

                        ols_model = sm.OLS(y_train_ols, X_train_ols).fit()
                        y_pred_ols = ols_model.predict(X_test_ols)
                        mse_ols = mean_squared_error(y_test_ols, y_pred_ols)
                        all_ols_mse_folds.append(mse_ols)
                        all_ols_beta_coefficients.append(ols_model.params.values)

                # Select the least 3 MSEs across all iterations and folds for OLS
                top_3_indices_ols = np.argsort(all_ols_mse_folds)[:3]
                top_3_mse_ols = [all_ols_mse_folds[i] for i in top_3_indices_ols]
                avg_top_3_mse_ols = np.mean(top_3_mse_ols)
                top_3_betas_ols = [all_ols_beta_coefficients[i] for i in top_3_indices_ols]
                avg_top_3_betas_ols = np.mean(top_3_betas_ols, axis=0)

                model_attributes.append({
                    'Dataset': key, 'Clusters': use_clusters, 'Model': 'OLS', 'K': k, 
                    'Avg Beta Estimates': pd.Series(avg_top_3_betas_ols, index=ols_model.params.index),
                    'Avg Top 3 MSE': avg_top_3_mse_ols,
                    'R^2': ols_model.rsquared,
                    'Residuals': ols_model.resid
                })
        
        results[key] = model_attributes

    return results

# Function to plot wls and ols with dropdowns
def plot_wls_ols_with_dropdowns(results_dict):
    variables = set()
    k_values = set()
    for key in results_dict:
        for entry in results_dict[key]:
            if 'Avg Beta Estimates' in entry:
                variables.update(entry['Avg Beta Estimates'].keys())
            k_values.add(entry['K'])
    
    variables = sorted(variables)
    k_values = sorted(k_values)

    # Create subplots: 3 rows, 2 columns
    fig = make_subplots(rows=3, cols=2, subplot_titles=[
        'Beta Estimates (OLS)', 'Beta Estimates (WLS)',
        'Avg Top 3 MSE (OLS)', 'Avg Top 3 MSE (WLS)',
        '$R^2 (OLS)$', '$R^2 (WLS)$'
    ], vertical_spacing=0.1, horizontal_spacing=0.1)

    def add_traces_to_fig(df, title_suffix):
        for var in variables:
            for cluster_status in [True, False]:
                for model in ['WLS', 'OLS']:
                    var_values = []
                    for entry in df:
                        if entry['Clusters'] == cluster_status and entry['Model'] == model:
                            k = entry['K']
                            if var in entry['Avg Beta Estimates']:
                                var_values.append(entry['Avg Beta Estimates'][var])
                            else:
                                var_values.append(0)  # Handle missing variable case

                    line_style = dict(dash='dash') if not cluster_status else dict()  # Dotted lines for Clusters=False
                    cluster_text = "Clusters=True" if cluster_status else "Clusters=False"
                    row, col = (1, 1) if model == 'OLS' else (1, 2)
                    fig.add_trace(go.Scatter(x=k_values, y=var_values, mode='lines+markers', name=f'{var} ({model}, {cluster_text}, {title_suffix})', line=line_style, visible=False), row=row, col=col)

        for cluster_status in [True, False]:
            for model in ['WLS', 'OLS']:
                r_squared_values = [entry['R^2'] for entry in df if entry['Clusters'] == cluster_status and entry['Model'] == model]
                avg_top3_mse_values = [entry['Avg Top 3 MSE'] for entry in df if entry['Clusters'] == cluster_status and entry['Model'] == model]
                
                line_style = dict(dash='dash') if not cluster_status else dict()  # Dotted lines for Clusters=False
                cluster_text = "Clusters=True" if cluster_status else "Clusters=False"
                row_r2, col_r2 = (3, 1) if model == 'OLS' else (3, 2)
                row_mse, col_mse = (2, 1) if model == 'OLS' else (2, 2)
                
                fig.add_trace(go.Scatter(x=k_values, y=avg_top3_mse_values, mode='lines+markers', name=f'Avg Top 3 MSE ({model}, {cluster_text}, {title_suffix})', line=line_style, visible=False), row=row_mse, col=col_mse)
                fig.add_trace(go.Scatter(x=k_values, y=r_squared_values, mode='lines+markers', name=f'$R^2 ({model}, {cluster_text}, {title_suffix})$', line=line_style, visible=False), row=row_r2, col=col_r2)

    add_traces_to_fig(results_dict['Nominal rates with clusters'], 'Nominal rates')
    add_traces_to_fig(results_dict['Nominal diff rates with clusters'], 'Nominal diff rates')
    add_traces_to_fig(results_dict['Nominal log with clusters'], 'Nominal log')

    # Update layout with dropdown and legend button
    fig.update_layout(
        title='Model Comparisons',
        showlegend=False,
        height=900, width=1400,  # Increase plot size
        updatemenus=[
            {
                'buttons': [
                    {
                        'label': 'Nominal rates',
                        'method': 'update',
                        'args': [
                            {'visible': [i < len(fig.data) // 3 for i in range(len(fig.data))]},
                            {'title': 'Model Comparisons for Nominal rates'}
                        ]
                    },
                    {
                        'label': 'Nominal diff rates',
                        'method': 'update',
                        'args': [
                            {'visible': [(len(fig.data) // 3 <= i < 2 * len(fig.data) // 3) for i in range(len(fig.data))]},
                            {'title': 'Model Comparisons for Nominal diff rates'}
                        ]
                    },
                    {
                        'label': 'Nominal log',
                        'method': 'update',
                        'args': [
                            {'visible': [i >= 2 * len(fig.data) // 3 for i in range(len(fig.data))]},
                            {'title': 'Model Comparisons for Nominal log'}
                        ]
                    },
                    {
                        'label': 'All',
                        'method': 'update',
                        'args': [
                            {'visible': [True for _ in range(len(fig.data))]},
                            {'title': 'Model Comparisons for All'}
                        ]
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.17,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            },
            {
                'buttons': [
                    {
                        'label': 'Show Legend',
                        'method': 'relayout',
                        'args': [{'showlegend': True}]
                    },
                    {
                        'label': 'Hide Legend',
                        'method': 'relayout',
                        'args': [{'showlegend': False}]
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.3,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top'
            }
        ]
    )

    # Initialize the plot with 'Nominal rates' visible
    for i in range(len(fig.data) // 3):
        fig.data[i].visible = True

    return fig


# Function to run wls regressions for all combinations of independent varibles (for multiple dataframes)
def run_regression_combinations(df, dependent_var, independent_vars, always_include=None, never_include=None, df_name='df', include_constant=False):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    county_unem = df.groupby('FIPS')[dependent_var].var()
    df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

    weight = df['weight']
    y = df[dependent_var]

    results = []

    if always_include is None:
        always_include = []
    if never_include is None:
        never_include = []

    independent_vars = [var for var in independent_vars if var not in never_include]

    for i in range(1, len(independent_vars) + 1):
        for combo in itertools.combinations(independent_vars, i):
            combo = list(always_include) + list(combo)
            X = df[combo]

            # Add a constant term if required
            if include_constant:
                X = sm.add_constant(X)
                combo = ['const'] + combo

            model = sm.WLS(y, X, weights=weight).fit()
            result = {
                'DataFrame': df_name,
                'Model': ', '.join(combo),
                'r-squared': model.rsquared,
                'Variables': '<br>'.join(
                    [f'{var}: {model.params[var]:.4f}' if isinstance(model.params[var], (int, float)) else f'{var}: {model.params[var]}'
                     if var in model.params.index else f'{var}: nan' for var in combo])  # Format variables and their values for hover text
            }
            for var in combo:
                result[var] = model.params.get(var, np.nan)
            results.append(result)

    results_df = pd.DataFrame(results)
    return results_df

# Function to plot the wls regressions from the multiple dataframes
def plot_combined_results(results_dfs, file_name):
    combined_results = pd.concat(results_dfs)
    fig = go.Figure()

    for df_name, group in combined_results.groupby('DataFrame'):
        for column in group.drop(columns=['DataFrame', 'Model', 'r-squared', 'Variables']).columns:
            fig.add_trace(go.Scatter(
                x=group['Model'],
                y=group[column],
                mode='markers',
                name=f'{df_name}: {column}',
                text=group.apply(lambda row: f"Variables:<br>{row['Variables']}<br>R-squared: {row['r-squared']:.4f}<br>Value: {row[column]:.4f}", axis=1),  # Set hover text with variables, R-squared, and their values
                hoverinfo='text'  # Display hover text
            ))

    fig.update_layout(
        title='Combined Regression Results',
        xaxis_title='Models',
        yaxis_title='Values',
        legend_title='Variables',
        autosize=False,
        width=1600,  # Adjust the width
        height=800,  # Adjust the height
        margin=dict(
            l=100,
            r=100,
            b=200,  # Adjust bottom margin to accommodate long x-axis labels
            t=100
        )
    )

    # Save the plot as an HTML file
    pio.write_html(fig, file=file_name, auto_open=False)
    
    
# Function to plot wls regressions combinations with k-folds
def run_regression_combinations_kfold(df, dependent_var, independent_vars, always_include=None, never_include=None, 
                                df_name='df', include_constant=False, n_splits=10, random_state=None):
    np.random.seed(random_state)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    county_unem = df.groupby('FIPS')[dependent_var].var()
    df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

    weight = df['weight']
    y = df[dependent_var]

    results = []

    if always_include is None:
        always_include = []
    if never_include is None:
        never_include = []

    independent_vars = [var for var in independent_vars if var not in never_include]

    for i in range(1, len(independent_vars) + 1):
        for combo in itertools.combinations(independent_vars, i):
            combo = list(always_include) + list(combo)
            X = df[combo]

            if include_constant:
                X = sm.add_constant(X)
                combo = ['const'] + combo

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            mse_list = []
            r_squared_list = []
            beta_estimates = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                weights_train = weight.iloc[train_index]

                model = sm.WLS(y_train, X_train, weights=weights_train).fit()
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                mse_list.append(mse)
                r_squared = model.rsquared
                r_squared_list.append(r_squared)
                beta_estimates.append(model.params)

            top_3_mse = sorted(mse_list)[:3]
            avg_top_3_mse = np.mean(top_3_mse)
            avg_beta_estimates = np.mean(beta_estimates, axis=0)
            avg_r_squared = np.mean(r_squared_list)

            result = {
                'DataFrame': df_name,
                'Model': ', '.join(combo),
                'r-squared': avg_r_squared,
                'avg_top_3_mse': avg_top_3_mse,
                'Variables': '<br>'.join(
                    [f'{combo[idx]}: {avg_beta_estimates[idx]:.4f}' for idx in range(len(combo))])
            }
            for idx, var in enumerate(combo):
                result[var] = avg_beta_estimates[idx]
            results.append(result)

    results_df = pd.DataFrame(results)
    return results_df


# Function to plot the combined wls regresions results from k-fold
def plot_combined_results_kfold(results_dfs, file_name):
    combined_results = pd.concat(results_dfs)
    fig = go.Figure()

    for df_name, group in combined_results.groupby('DataFrame'):
        for column in group.drop(columns=['DataFrame', 'Model', 'r-squared', 'Variables', 'avg_top_3_mse']).columns:
            fig.add_trace(go.Scatter(
                x=group['Model'],
                y=group[column],
                mode='markers',
                name=f'{df_name}: {column}',
                text=group.apply(lambda row: f"Variables:<br>{row['Variables']}<br>R-squared: {row['r-squared']:.4f}<br>Avg Top 3 MSE: {row['avg_top_3_mse']:.4f}<br>Value: {row[column]:.4f}", axis=1),  # Set hover text with variables, R-squared, Avg Top 3 MSE, and their values
                hoverinfo='text'  # Display hover text
            ))

    fig.update_layout(
        title='Combined Regression Results',
        xaxis_title='Models',
        yaxis_title='Values',
        legend_title='Variables',
        autosize=False,
        width=1600,  # Adjust the width
        height=800,  # Adjust the height
        margin=dict(
            l=100,
            r=100,
            b=200,  # Adjust bottom margin to accommodate long x-axis labels
            t=100
        )
    )

    # Save the plot as an HTML file
    pio.write_html(fig, file=file_name, auto_open=False)
    

# Function to fit ols or wls using k-fold
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout  # Using pygraphviz layout

# def ols_wls_combinations_kfold(df, dependent_var, independent_vars, model_type='ols', 
#                                       always_include=None, never_include=None, df_name='df', 
#                                       include_constant=False, n_splits=10, random_state=None):
#     np.random.seed(random_state)
#     df = df.replace([np.inf, -np.inf], np.nan).dropna()

#     y = df[dependent_var]

#     if model_type not in ['ols', 'wls']:
#         raise ValueError("model_type must be either 'ols' or 'wls'")

#     if model_type == 'wls':
#         # Calculate weights for WLS
#         county_unem = df.groupby('FIPS')[dependent_var].var()
#         df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
#         df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

#     results = []
#     residuals_dict = {var: [] for var in [dependent_var] + independent_vars}

#     if always_include is None:
#         always_include = []
#     if never_include is None:
#         never_include = []

#     independent_vars = [var for var in independent_vars if var not in never_include]

#     for i in range(1, len(independent_vars) + 1):
#         for combo in itertools.combinations(independent_vars, i):
#             combo = list(always_include) + list(combo)
#             X = df[combo]

#             if include_constant:
#                 X = sm.add_constant(X)
#                 combo = ['const'] + combo

#             kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#             mse_list = []
#             r_squared_list = []
#             beta_estimates = []
#             fold_residuals = {var: [] for var in [dependent_var] + combo if var != 'const'}

#             for train_index, test_index in kf.split(X):
#                 X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
#                 y_train, y_test = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index].reset_index(drop=True)

#                 if model_type == 'ols':
#                     model = sm.OLS(y_train, X_train).fit()
#                 elif model_type == 'wls':
#                     weights = df['weight'].iloc[train_index].reset_index(drop=True)
#                     model = sm.WLS(y_train, X_train, weights=weights).fit()

#                 y_pred = model.predict(X_test)
#                 mse = np.mean((y_test - y_pred) ** 2)
#                 mse_list.append(mse)
#                 r_squared = model.rsquared
#                 r_squared_list.append(r_squared)
#                 beta_estimates.append(model.params)
                
#                 residuals_y = y_test - y_pred
#                 fold_residuals[dependent_var].extend(residuals_y)

#                 # Calculate and store residuals for independent variables
#                 for var in combo:
#                     if var != 'const':
#                         residuals_X = X_test[var] - model.predict(X_test)
#                         fold_residuals[var].extend(residuals_X)

#             top_3_mse = sorted(mse_list)[:3]
#             avg_top_3_mse = np.mean(top_3_mse)
#             avg_beta_estimates = np.mean(beta_estimates, axis=0)
#             avg_r_squared = np.mean(r_squared_list)

#             result = {
#                 'DataFrame': df_name,
#                 'Model': ', '.join(combo),
#                 'r-squared': avg_r_squared,
#                 'avg_top_3_mse': avg_top_3_mse,
#                 'Variables': '<br>'.join(
#                     [f'{combo[idx]}: {avg_beta_estimates[idx]:.4f}' for idx in range(len(combo))])
#             }
#             for idx, var in enumerate(combo):
#                 result[var] = avg_beta_estimates[idx]
#             results.append(result)

#             # Store residuals for the model
#             for var in fold_residuals:
#                 if var in residuals_dict:
#                     residuals_dict[var].extend(fold_residuals[var])
#                 else:
#                     residuals_dict[var] = fold_residuals[var]

#     # Make sure all lists in residuals_dict have the same length
#     max_length = max(len(lst) for lst in residuals_dict.values())
#     for var in residuals_dict:
#         if len(residuals_dict[var]) < max_length:
#             residuals_dict[var].extend([np.nan] * (max_length - len(residuals_dict[var])))

#     results_df = pd.DataFrame(results)
#     return results_df, residuals_dict

# Function to perform wls and ols regressions (a shorter version)
# To reduce lines of code
def perform_regression(df, dv, iv, model_type, ai, ni, df_name, include_constant, n_splits, random_state):
    return ols_wls_combinations_kfold(
        df, dv, iv, 
        model_type=model_type, 
        always_include=ai, 
        never_include=ni, 
        df_name=df_name, 
        include_constant=include_constant, 
        n_splits=n_splits, 
        random_state=random_state
    )



##################################################################
#updated portion 07092024

def build_skeleton(df, undirected_graph):
    # Iterate over each pair of variables (columns) in the DataFrame
    for i, var1 in enumerate(df.columns):
        for var2 in df.columns[i + 1:]:
            corr_value = abs(df[var1].corr(df[var2]))
            if corr_value > 0.01:
                # Add an undirected edge between var1 and var2 if correlation exceeds 0.5
                undirected_graph[var1].append(var2)
                undirected_graph[var2].append(var1)
#                 print(f"Adding edge between {var1} and {var2} with correlation {corr_value:.2f}")
    return undirected_graph

def graph_undirected_DAG(undirected_graph, title, ax):
    graph = nx.Graph()
    for node, edges in undirected_graph.items():
        for edge in edges:
            graph.add_edge(node, edge)
    pos = graphviz_layout(graph, prog='neato')
    nx.draw(graph, pos, ax=ax, node_color="C0", node_size=1500, with_labels=True, arrows=False, font_size=10, alpha=1, font_color="white")
    ax.set_title(title, fontsize=12)

def convert_to_directed(undirected_graph):
    dag = nx.DiGraph()
    for node, edges in undirected_graph.items():
        for edge in edges:
            dag.add_edge(node, edge)
    return dag

def plot_dag_for_dataframe(df, exclude_columns=None, ax=None, title=None):
    if exclude_columns is not None:
        drop_columns = [col for col in exclude_columns if col in df.columns]
        df = df.drop(columns=drop_columns)

    undirected_graph = {key: [] for key in df.columns}
    undirected_graph = build_skeleton(df, undirected_graph)
    # graph_undirected_DAG(undirected_graph, title=f"Undirected Graph: {title}", ax=ax)
    dag = convert_to_directed(undirected_graph)

    pos = graphviz_layout(dag, prog='dot')
    nx.draw(dag, pos, ax=ax, node_color="C0", node_size=1500, with_labels=True, arrows=True, font_size=10, alpha=1, font_color="black")
    ax.set_title(f"DAG: {title}", fontsize=12)
    ax.axis("off")

def plot_dags_in_grid(dag_dict, include_columns=None, exclude_columns=None):
    if include_columns is not None:
        include_columns = set(include_columns)
    else:
        include_columns = set()
    if exclude_columns is None:
        exclude_columns = []

    num_plots = len(dag_dict)
    num_cols = 2  # Number of columns in the grid layout
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axs = axs.flatten()

    for i, (dag_name, dag_df) in enumerate(dag_dict.items()):
        ax = axs[i]

        if include_columns:
            columns_to_plot = [col for col in dag_df.columns if col in include_columns]
        else:
            columns_to_plot = [col for col in dag_df.columns if col not in exclude_columns]

        plot_dag_for_dataframe(dag_df[columns_to_plot], ax=ax, title=dag_name)

    for j in range(i + 1, num_cols * num_rows):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
    

#Function to fit single regression kfolds for plotting dags    
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

def ols_wls_single_regression_kfold(df, dependent_var, independent_vars, model_type='ols', 
                                    always_include=None, never_include=None, df_name='df', 
                                    include_constant=False, n_splits=10, random_state=None):
    np.random.seed(random_state)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    y = df[dependent_var]

    if model_type not in ['ols', 'wls']:
        raise ValueError("model_type must be either 'ols' or 'wls'")

    if model_type == 'wls':
        # Calculate weights for WLS
        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

    results = []
    residuals_dict = {var: [] for var in [dependent_var] + independent_vars}

    if always_include is None:
        always_include = []
    if never_include is None:
        never_include = []

    independent_vars = [var for var in independent_vars if var not in never_include]

    # Only use the predefined combination of variables
    combo = list(always_include) + independent_vars
    X = df[combo]

    if include_constant:
        X = sm.add_constant(X)
        combo = ['const'] + combo

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mse_list = []
    r_squared_list = []
    beta_estimates = []
    fold_residuals = {var: [] for var in [dependent_var] + combo if var != 'const'}

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
        y_train, y_test = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index].reset_index(drop=True)

        if model_type == 'ols':
            model = sm.OLS(y_train, X_train).fit()
        elif model_type == 'wls':
            weights = df['weight'].iloc[train_index].reset_index(drop=True)
            model = sm.WLS(y_train, X_train, weights=weights).fit()

        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        mse_list.append(mse)
        r_squared = model.rsquared
        r_squared_list.append(r_squared)
        beta_estimates.append(model.params)
        
        residuals_y = y_test - y_pred
        fold_residuals[dependent_var].extend(residuals_y)

        # Calculate and store residuals for independent variables
        for var in combo:
            if var != 'const':
                residuals_X = X_test[var] - model.predict(X_test)
                fold_residuals[var].extend(residuals_X)

    top_3_mse = sorted(mse_list)[:3]
    avg_top_3_mse = np.mean(top_3_mse)
    avg_beta_estimates = np.mean(beta_estimates, axis=0)
    avg_r_squared = np.mean(r_squared_list)

    result = {
        'DataFrame': df_name,
        'Model': ', '.join(combo),
        'r-squared': avg_r_squared,
        'avg_top_3_mse': avg_top_3_mse,
        'Variables': '<br>'.join(
            [f'{combo[idx]}: {avg_beta_estimates[idx]:.4f}' for idx in range(len(combo))]),
        'Model_Type': model_type
    }
    for idx, var in enumerate(combo):
        result[var] = avg_beta_estimates[idx]
    results.append(result)

    # Store residuals for the model
    for var in fold_residuals:
        if var in residuals_dict:
            residuals_dict[var].extend(fold_residuals[var])
        else:
            residuals_dict[var] = fold_residuals[var]

    # Make sure all lists in residuals_dict have the same length
    max_length = max(len(lst) for lst in residuals_dict.values())
    for var in residuals_dict:
        if len(residuals_dict[var]) < max_length:
            residuals_dict[var].extend([np.nan] * (max_length - len(residuals_dict[var])))

    results_df = pd.DataFrame(results)
    residuals_df = pd.DataFrame(residuals_dict)
    return results_df, residuals_df


import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

def ols_wls_combinations_kfold(df, dependent_var, independent_vars, model_type='ols', 
                               always_include=None, never_include=None, df_name='df', 
                               include_constant=False, n_splits=10, random_state=None):
    np.random.seed(random_state)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    y = df[dependent_var]

    if model_type not in ['ols', 'wls']:
        raise ValueError("model_type must be either 'ols' or 'wls'")

    if model_type == 'wls':
        # Calculate weights for WLS
        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem.get(x, np.nan))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

    results = []
    residuals_dict = {var: [] for var in [dependent_var] + independent_vars}

    if always_include is None:
        always_include = []
    if never_include is None:
        never_include = []

    independent_vars = [var for var in independent_vars if var not in never_include]

    for i in range(1, len(independent_vars) + 1):
        for combo in itertools.combinations(independent_vars, i):
            combo = list(always_include) + list(combo)
            X = df[combo]

            if include_constant:
                X = sm.add_constant(X)
                combo = ['const'] + combo

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            mse_list = []
            r_squared_list = []
            beta_estimates = []
            fold_residuals = {var: [] for var in [dependent_var] + combo if var != 'const'}

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
                y_train, y_test = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index].reset_index(drop=True)

                if model_type == 'ols':
                    model = sm.OLS(y_train, X_train).fit()
                elif model_type == 'wls':
                    weights = df['weight'].iloc[train_index].reset_index(drop=True)
                    model = sm.WLS(y_train, X_train, weights=weights).fit()

                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                mse_list.append(mse)
                r_squared = model.rsquared
                r_squared_list.append(r_squared)
                beta_estimates.append(model.params)
                
                residuals_y = y_test - y_pred
                fold_residuals[dependent_var].extend(residuals_y)

                # Calculate and store residuals for independent variables
                for var in combo:
                    if var != 'const':
                        residuals_X = X_test[var] - model.predict(X_test)
                        fold_residuals[var].extend(residuals_X)

            top_3_mse = sorted(mse_list)[:3]
            avg_top_3_mse = np.mean(top_3_mse)
            avg_beta_estimates = np.mean(beta_estimates, axis=0)
            avg_r_squared = np.mean(r_squared_list)

            result = {
                'DataFrame': df_name,
                'Model': ', '.join(combo),
                'r-squared': avg_r_squared,
                'avg_top_3_mse': avg_top_3_mse,
                'Variables': '<br>'.join(
                    [f'{combo[idx]}: {avg_beta_estimates[idx]:.4f}' for idx in range(len(combo))]),
                'Model_Type': model_type
            }
            for idx, var in enumerate(combo):
                result[var] = avg_beta_estimates[idx]
            results.append(result)

            # Store residuals for the model
            for var in fold_residuals:
                if var in residuals_dict:
                    residuals_dict[var].extend(fold_residuals[var])
                else:
                    residuals_dict[var] = fold_residuals[var]

    # Make sure all lists in residuals_dict have the same length
    max_length = max(len(lst) for lst in residuals_dict.values())
    for var in residuals_dict:
        if len(residuals_dict[var]) < max_length:
            residuals_dict[var].extend([np.nan] * (max_length - len(residuals_dict[var])))

    results_df = pd.DataFrame(results)
    return results_df, residuals_dict


import plotly.express as px
#Function to plot regressions with shapes
def plot_single_results_kfold(results_dfs, file_name):
    combined_results = pd.concat(results_dfs)
    fig = go.Figure()

    # Define shapes for each variable
    variable_shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down']
    variables = combined_results['Variables'].unique()
    variable_shape_map = {variable: shape for variable, shape in zip(variables, variable_shapes)}

    # Define a color palette for variables
    color_palette = px.colors.qualitative.Plotly[:len(variables)]
    variable_color_map = {variable: color for variable, color in zip(variables, color_palette)}

    for df_name, group in combined_results.groupby('DataFrame'):
        for column in group.drop(columns=['DataFrame', 'Model', 'r-squared', 'Variables', 'avg_top_3_mse', 'Model_Type']).columns:
            for variable in variables:
                variable_group = group[group['Variables'] == variable]
                if not variable_group.empty:
                    model_type = variable_group['Model_Type'].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=variable_group['Model'],
                        y=variable_group[column],
                        mode='markers',
                        name=f'{df_name}: {column}',
                        text=variable_group.apply(lambda row: f"Model Type: {model_type}<br>Variables:<br>{row['Variables']}<br>R-squared: {row['r-squared']:.4f}<br>Avg Top 3 MSE: {row['avg_top_3_mse']:.4f}<br>Value: {row[column]:.4f}", axis=1),  # Set hover text with model type, variables, R-squared, Avg Top 3 MSE, and their values
                        hoverinfo='text',  # Display hover text
                        marker=dict(
                            symbol=variable_shape_map[variable],  # Use shape based on variable
                            color=variable_color_map[variable]  # Use color based on variable
                        )
                    ))

    fig.update_layout(
        title='Combined Regression Results',
        xaxis_title='Models',
        yaxis_title='Values',
        legend_title='Variables',
        autosize=False,
        width=1600,  # Adjust the width
        height=800,  # Adjust the height
        margin=dict(
            l=100,
            r=100,
            b=200,  # Adjust bottom margin to accommodate long x-axis labels
            t=100
        )
    )

    # Save the plot as an HTML file
    pio.write_html(fig, file=file_name, auto_open=False)
