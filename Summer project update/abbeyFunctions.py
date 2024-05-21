
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

# Define a function to plot a DAG
def graph_DAG(edges, df, title=""):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.keys() if key not in edge]
        controls = list(set(controls))
        keep_controls = []
        for control in controls:
            control_edges = [ctrl_edge for ctrl_edge in edges if control == ctrl_edge[0]]
            if (control, edge[1]) in control_edges:
                print('keep control:', control)
                keep_controls.append(control)
        print(edge, keep_controls)
        pcorr = df[[edge[0], edge[1]] + keep_controls].pcorr()
        edge_labels[edge] = str(round(pcorr[edge[0]].loc[edge[1]], 2))
    graph.add_edges_from(edges)
    color_map = ['grey' for _ in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    plt.tight_layout()
    pos = graphviz_layout(graph)#, "neato", None)

    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                      arrows=True, font_color='black', font_size=26, alpha=1, width=1,
                      edge_color='C1',
                      arrowstyle='Fancy, head_length=3, head_width=1.5, tail_width=.1',
                     connectionstyle='arc3, rad = 0.05', ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)

    
    
    
    
    
# function for creating DAGs from a dataframe that has clusters

def create_cluster_dags(df):
    grouped = df.groupby('clusters')

    # Initialize an empty dictionary to store the DataFrames for each cluster
    cluster_dataframes = {}

    # Iterate over the groups and create DataFrames for each cluster
    for cluster, group in grouped:
        cluster_dataframes[cluster] = group.drop(['clusters'], axis=1)

    # List of DataFrames for each cluster
    cluster_dataframes_list = [cluster_dataframes[i] for i in range(1, 5)]  # Assuming four clusters

    return cluster_dataframes_list

def determine_p_value(n):
    # If the number of observations is 800 and above, use a p-value of 0.05, otherwise use 0.10
    return 0.05 if n >= 800 else 0.1

def graph_DAG_cluster(edges, df, title="", algorithm="parallel", ax=None, sig_vals=[0.05, 0.01, 0.001], pp=None):
    graph = nx.DiGraph()

    def build_edge_labels(edges, df, sig_vals):
        edge_labels = {}
        for edge in edges:
            controls = [key for key in df.keys() if key not in edge]
            controls = list(set(controls))
            keep_controls = []
            for control in controls:
                control_edges = [ctrl_edge for ctrl_edge in edges if control == ctrl_edge[0]]
                if (control, edge[1]) in control_edges:
                    keep_controls.append(control)

            pcorr = df.partial_corr(x=edge[0], y=edge[1], covar=keep_controls, method="pearson")
            label = str(round(pcorr["r"][0], 2))
            pvalue = pcorr["p-val"][0]

            for sig_val in sig_vals:
                if pvalue < sig_val:
                    label = label + "*"

            edge_labels[edge] = label
        return edge_labels

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))

    graph.add_edges_from(edges)
    color_map = ["grey" for g in graph]

    graph.nodes()
    plt.tight_layout()
    pos = graphviz_layout(graph)

    edge_labels = build_edge_labels(edges, df, sig_vals=sig_vals)

    edge_labels2 = []
    for u, v, d in graph.edges(data=True):
        if pos[u][0] > pos[v][0]:
            if (v, u) in edge_labels.keys():
                edge_labels2.append(((u, v,), f'{edge_labels[u, v]}\n\n\n{edge_labels[(v, u)]}'))
        if (v, u) not in edge_labels.keys():
            edge_labels2.append(((u, v,), f'{edge_labels[(u, v)]}'))
    edge_labels = dict(edge_labels2)

    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1000,
                     with_labels=True, arrows=True,
                     font_color="black",
                     font_size=15, alpha=1,
                     width=1, edge_color="C1",
                     arrowstyle="Fancy, head_length=2, head_width=1.5, tail_width=.1",
                     connectionstyle='arc3, rad = 0.05',
                     ax=ax)
    nx.draw_networkx_edge_labels(graph,
                                 pos=pos,
                                 edge_labels=edge_labels,
                                 font_color='green',
                                 font_size=20,
                                 ax=ax)

    # Add title to the DAG
    ax.set_title(title)

    # Save to PDF if a PdfPages object is provided
    if pp:
        pp.savefig(ax.figure, bbox_inches='tight')


        
        
        
# function to get the list of edges

def get_edges_dict(dataframes):
    edges_dct = {}
    for idx, dataframe in enumerate(dataframes):
        # Get clustered dataframes for the current dataframe
        clustered_dfs = create_cluster_dags(dataframe)
        edges_dct[idx] = {}
        # Iterate over clustered dataframes and determine p-values dynamically
        for cluster_idx, cluster_df in enumerate(clustered_dfs):
            n = cluster_df.shape[0]  # Number of observations for the current cluster
            p_value = determine_p_value(n)

            # Iterate over different data for PC algorithm
            algorithm = "orig"
            c = PC(cluster_df)
            model = c.estimate(return_type='pdag', variant=algorithm, significance_level=p_value, ci_test='pearsonr')
            edges = model.edges
            edges = list(set([tuple(sorted(edge)) for edge in edges]))
            edges_dct[idx][cluster_idx] = edges
    return edges_dct




# function to get the list of shared column edges

def get_col_shared_edges(edges_dct):
    cols = list(edges_dct.keys())
    rows = list(edges_dct[0].keys())

    col_shared_edges = {}
    row_shared_edges = {}
    for col in cols:
        for row, edges in edges_dct[col].items():
            if row == 0: col_shared_edges[col] = copy.copy(edges)
            col_shared_edges[col]  = [edge for edge in edges if edge in col_shared_edges[col] ]

    return col_shared_edges





# function to get the list of shared row edges

def get_row_shared_edges(edges_dct):
    cols = list(edges_dct.keys())
    rows = list(edges_dct[0].keys())

    row_shared_edges = {}
    for row in rows:
        row_shared_edges[row] = {}
        for col in cols:
            if col == 0: row_shared_edges[row] = edges_dct[col][row]
            row_shared_edges[row] = [edge for edge in edges_dct[col][row] if edge in row_shared_edges[row]]

    return row_shared_edges






# function for plotting the shared row or column edges

# def plot_shared_edges(shared_edges, filename=None):
#     for key, lst in shared_edges.items():
#         fig, ax = plt.subplots(figsize=(6, 6))
#         G = nx.from_edgelist(lst)
#         color_map = ["grey" for _ in G]
#         plt.tight_layout()
#         pos = graphviz_layout(G)
#         nx.draw_networkx(G, pos, node_color=color_map, node_size=2500,
#                          font_color="black",
#                          font_size=26, alpha=1,
#                          width=1, edge_color="C1",
#                          connectionstyle='arc3, rad = 0.05',
#                          ax=ax)
#         if filename:
#             plt.savefig(filename, format='pdf')
#         else:
#             plt.show()
#         plt.close()
        
        
# def plot_shared_edges(shared_edges_dfs, folder):
#     # Create a subfolder for the shared edges plots
#     shared_edges_folder = os.path.join(folder, 'shared_edges_plots')
#     try:
#         os.mkdir(shared_edges_folder)
#     except FileExistsError:
#         pass

#     for idx, shared_edges in enumerate(shared_edges_dfs):
#         for key, lst in shared_edges.items():
#             fig, ax = plt.subplots(figsize=(6, 6))
#             G = nx.from_edgelist(lst)
#             color_map = ["grey" for _ in G]
#             plt.tight_layout()
#             pos = graphviz_layout(G)
#             nx.draw_networkx(G, pos, node_color=color_map, node_size=2500,
#                              font_color="black",
#                              font_size=26, alpha=1,
#                              width=1, edge_color="C1",
#                              connectionstyle='arc3, rad = 0.05',
#                              ax=ax)
#             plt.savefig(os.path.join(shared_edges_folder, f'shared_edges_plot_{idx}.pdf'))
#             plt.close()


# def plot_shared_edges(shared_edges, folder):
#     for key, lst in shared_edges.items():
#         fig, ax = plt.subplots(figsize=(6, 6))
#         G = nx.from_edgelist(lst)
#         color_map = ["grey" for _ in G]
#         plt.tight_layout()
#         pos = graphviz_layout(G)
#         nx.draw_networkx(G, pos, node_color=color_map, node_size=2500,
#                          font_color="black",
#                          font_size=26, alpha=1,
#                          width=1, edge_color="C1",
#                          connectionstyle='arc3, rad = 0.05',
#                          ax=ax)


def plot_shared_col_edges(shared_edges, name, folder):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    
    for key, lst in shared_edges.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        G = nx.from_edgelist(lst)
        color_map = ["grey" for _ in G]
        plt.tight_layout()
        pos = graphviz_layout(G)
        nx.draw_networkx(G, pos, node_color=color_map, node_size=2500,
                         font_color="black",
                         font_size=26, alpha=1,
                         width=1, edge_color="C1",
                         connectionstyle='arc3, rad = 0.05',
                         ax=ax)
        pdf_file_path = os.path.join(folder, f'{name}_col_{key}.pdf')
        with PdfPages(pdf_file_path) as pdf:
            pdf.savefig(fig)
        plt.close()

        
def plot_shared_row_edges(shared_edges, name, folder):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    
    for key, lst in shared_edges.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        G = nx.from_edgelist(lst)
        color_map = ["grey" for _ in G]
        plt.tight_layout()
        pos = graphviz_layout(G)
        nx.draw_networkx(G, pos, node_color=color_map, node_size=2500,
                         font_color="black",
                         font_size=26, alpha=1,
                         width=1, edge_color="C1",
                         connectionstyle='arc3, rad = 0.05',
                         ax=ax)
        pdf_file_path = os.path.join(folder, f'{name}_row_{key}.pdf')
        with PdfPages(pdf_file_path) as pdf:
            pdf.savefig(fig)
        plt.close()

# function for plotting the shared row or column edges

def plot_shared_edges(shared_edges):
    for key, lst in shared_edges.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        G = nx.from_edgelist(lst)
        color_map = ["grey" for _ in G]
        plt.tight_layout()
        pos = graphviz_layout(G)
        nx.draw_networkx(G, pos, node_color=color_map, node_size=2500,
                         font_color="black",
                         font_size=26, alpha=1,
                         width=1, edge_color="C1",
                         connectionstyle='arc3, rad = 0.05',
                         ax=ax)
    plt.show()


#Function for performing hausman test with two mixed effect models    
    
def hausman_test(r_fe, r_me1, r_me2):
    b_fe = r_fe.params
    b_fe_cov = r_fe.cov

    b_me1 = r_me1.params
    b_me1_cov = r_me1.cov_params()

    b_me2 = r_me2.params
    b_me2_cov = r_me2.cov_params()

    common_coef = set(b_fe.index).intersection(b_me1.index).intersection(b_me2.index)
    b_diff_me1 = b_fe[common_coef].sub(b_me1[common_coef])
    b_diff_me2 = b_fe[common_coef].sub(b_me2[common_coef])

    df = len(b_diff_me1)

    b_cov_diff_me1 = b_fe_cov.loc[common_coef, common_coef].sub(b_me1_cov.loc[common_coef, common_coef])
    b_var_diff_me1 = pd.Series(np.diag(b_cov_diff_me1), index=[b_cov_diff_me1.index])

    b_cov_diff_me2 = b_fe_cov.loc[common_coef, common_coef].sub(b_me2_cov.loc[common_coef, common_coef])
    b_var_diff_me2 = pd.Series(np.diag(b_cov_diff_me2), index=[b_cov_diff_me2.index])

    stat_me1 = abs(b_diff_me1.T @ np.linalg.inv(b_cov_diff_me1) @ b_diff_me1)
    p_value_me1 = 1 - stats.chi2.cdf(stat_me1, df)

    stat_me2 = abs(b_diff_me2.T @ np.linalg.inv(b_cov_diff_me2) @ b_diff_me2)
    p_value_me2 = 1 - stats.chi2.cdf(stat_me2, df)
    
    h_results = {
        "Hausman_ME1": {
            "t": stat_me1,
            "p": p_value_me1
        },
        "Hausman_ME2": {
            "t": stat_me2,
            "p": p_value_me2
        }
    }
    
    return pd.DataFrame(h_results)
