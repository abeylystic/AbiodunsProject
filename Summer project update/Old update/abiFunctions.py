
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


# Function to build skeleton for undirected DAG
def build_skeleton(df, undirected_graph, p_val = 0.005):    
    def check_remaining_controls(control_vars, undirected_graph, x, y, controls_used):
        for c_var in control_vars:
            c_used = copy.copy(controls_used)
            if y in undirected_graph[x]:
                c_used.append(c_var)
                test = pg.partial_corr(data=df, x=x, y=y, covar=c_used, method="pearson")
                if test["p-val"].values[0] > p_val:
                    undirected_graph[x].remove(y)
                    break
                else:
                    remaining_controls = copy.copy(control_vars)
                    remaining_controls.remove(c_var)
                    check_remaining_controls(remaining_controls, undirected_graph, x, y, c_used)

    for x in df.columns:
        ys = undirected_graph[x]
        for y in ys[:]:  # Use a slice copy to avoid modifying the list during iteration
            if x != y:
                test = pg.partial_corr(data=df, x=x, y=y, covar=None, method="pearson")
                if test["p-val"].values[0] > p_val:
                    undirected_graph[x].remove(y)
                else:
                    control_vars = [z for z in df.columns if z != y and z != x]
                    check_remaining_controls(control_vars, undirected_graph, x, y, [])
    return undirected_graph

# Function to plot undirected DAG
def graph_undirected_DAG(undirected_graph, df, title="DAG Structure"):
    graph = nx.DiGraph()
    edges = []
    edge_labels = {}
    
    for key in undirected_graph:
        for key2 in undirected_graph[key]:
            if (key2, key) not in edges:
                edge = (key.replace(" ", "\n"), key2.replace(" ", "\n"))
                edges.append(edge)

    graph.add_edges_from(edges)
    color_map = ["C0" for _ in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    plt.tight_layout()
    pos = graphviz_layout(graph)

    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1500,
                     with_labels=True, arrows=False, font_size=20,
                     alpha=1, font_color="white", ax=ax)

    plt.axis("off")
    plt.savefig("g1.png", format="PNG")
    plt.show()
    

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
def analyze_wls_pooled_models(data_cluster_dict, dependent_var, k=None, shuffle=None, random_state=None, check_rank=None):
    # Placeholder lists to store performance metrics and model attributes
    mse_results = []
    model_attributes = []

    # Loop through the dataframes in the dictionary
    for key, df in data_cluster_dict.items():
        # Dropping rows with NaNs or infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate the weights based on the variance of the dependent variable for each county
        county_unem = df.groupby('FIPS')[dependent_var].var()
        df['weight'] = df['FIPS'].map(lambda x: 1 / county_unem[x])

        # Ensuring that the weights column has no NaNs or infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['weight'])

        # Define the dependent variable
        y = df[dependent_var]
        
        # Define the independent variables and exclude 'FIPS' and 'TimePeriod'
        X = df.drop(columns=[dependent_var, 'FIPS', 'TimePeriod'])

        weights = df['weight']

        for use_clusters in [True, False]:
            X_filtered = X.drop(columns=[col for col in X.columns if 'clusters' in col]) if not use_clusters else X

            # Checking for multicollinearity using VIF
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_filtered.columns
            vif_data["VIF"] = [variance_inflation_factor(X_filtered.dropna().values, i) for i in range(len(X_filtered.columns))]

            # Setting up k-fold cross-validation
            kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

            # Placeholder lists to store MSE for each fold
            mse_folds = []

            # Performing k-fold cross-validation
            for train_index, test_index in kf.split(df):
                # Split the data into training and validation sets
                X_train, X_test = X_filtered.iloc[train_index], X_filtered.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                weights_train, weights_test = weights.iloc[train_index], weights.iloc[test_index]

                # Fit the WLS models with the weights
                model = sm.WLS(y_train, X_train, weights=weights_train).fit()

                # Making predictions on the validation set
                y_pred = model.predict(X_test)

                # Calculating the mean squared error for each fold
                mse_folds.append(mean_squared_error(y_test, y_pred))

            # Computing the average mean squared error across all folds for this configuration
            avg_mse = np.mean(mse_folds)

            # Storing the results
            mse_results.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'WLS', 'Avg MSE': avg_mse, 'MSE Folds': mse_folds, 'R^2': model.rsquared})

            # Collecting WLS model attributes
            model_attributes.append({
                'Dataset': key,
                'Clusters': use_clusters,
                'Model': 'WLS',
                'Beta Estimates': pd.Series(model.params),
                'R^2': model.rsquared
            })

            # Convert y_train and y_test to DataFrame and set 2-level MultiIndex
            y_train_pooled = y_train.to_frame().set_index([df.iloc[train_index].index, df.iloc[train_index]['TimePeriod']])
            y_test_pooled = y_test.to_frame().set_index([df.iloc[test_index].index, df.iloc[test_index]['TimePeriod']])
        
            # Set 2-level MultiIndex for X_train and X_test
            X_train_pooled = X_train.set_index([df.iloc[train_index].index, df.iloc[train_index]['TimePeriod']])
            X_test_pooled = X_test.set_index([df.iloc[test_index].index, df.iloc[test_index]['TimePeriod']])
        
            # Fitting the pooled OLS models
            pooled_model = PooledOLS(y_train_pooled, X_train_pooled, check_rank=check_rank).fit()

            # Making predictions on the validation set
            y_pred_pooled = pooled_model.predict(X_test_pooled)

            # Calculating the mean squared error for the pooled OLS model
            mse_pooled = mean_squared_error(y_test_pooled, y_pred_pooled)

            # Storing the results for pooled OLS
            mse_results.append({'Dataset': key, 'Clusters': use_clusters, 'Model': 'PooledOLS', 'Avg MSE': mse_pooled, 'MSE Folds': mse_folds, 'R^2': pooled_model.rsquared})

            # Collecting PooledOLS model attributes
            model_attributes.append({
                'Dataset': key,
                'Clusters': use_clusters,
                'Model': 'PooledOLS',
                'Beta Estimates': pd.Series(pooled_model.params),
                'R^2': pooled_model.rsquared
            })

    # Create a DataFrame for beta estimates, MSEs, and R^2
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
