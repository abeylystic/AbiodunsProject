
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

            if 'Nominal rates' in key:
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
