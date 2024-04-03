#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install datapungibea
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import os

import datapungibea as dpb
key = '1FD5DC35-4854-4CE8-8D43-B36065C37041'
data = dpb.data(key) 
data


# In[2]:


# data.Regional()


# In[3]:


# # Collect county level variables
# for_county_index = data.Regional(GeoFips = "COUNTY",
#               LineCode = "1",
#               TableName = "CAGDP9", 
#               Year = "2020")
# for_county_index


# In[4]:


# counties = for_county_index["GeoFips"]
# counties


# In[5]:


# for_GDP_components = data.Regional(GeoFips = "01001",
#               LineCode = "ALL",
#               TableName = "CAGDP9", 
#               Year = "2010")
# for_GDP_components


# In[6]:


# GDP_components = for_GDP_components[["Code", "Description"]]
# GDP_components


# In[7]:


# import time
# import random

# for row in GDP_components.iterrows():
#     full_code, description = row[1]
#     table_name, code = full_code.split("-")
   
#     try:
#         GDP_data_dct[description] = data.Regional(GeoFips = "COUNTY",
#               LineCode = code,
#               TableName = table_name, 
#               Year = "ALL")
#         print(full_code +": " + description + " downloaded")
#     except:
#         print(full_code +": Error downloading " + description)
#     time.sleep(10)


# In[ ]:





# In[8]:


# import os
# GDP_data_dct = {}
# folder1 = "Data"
# folder2 = "CountyGDP"
# try:
#     os.mkdir(folder1)
#     os.mkdir(folder1 + "/" + folder2)
# except:
#     try: 
#         os.mkdir(folder1 + "/" + folder2)
#     except:
#         pass
# # for key, val in GDP_data_dct.items():
# #     val.to_csv(folder1 + "/" + folder2 + "/" + key.replace("/","") + ".csv")

# for row in GDP_components.iterrows():
#     full_code, description = row
#     description = description[1]
#     # somehow, space added in very last character of description
#     GDP_data_dct[description] = pd.read_csv("Data/CountyGDP/" + description[:-1].replace("/","") + ".csv")
# GDP_data_dct
# # GDP_components


# In[9]:


#import data and convert datatype to int64
data = pd.read_csv('AbiData.csv')
# data.astype('int64')
# data = data.apply(pd.to_numeric, errors='coerce')
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.to_numeric(data[column].str.replace(',', ''), errors='coerce').astype('Int64')
        


# In[10]:


#set index as GeoFips and TimePeriod
data = data.set_index(["GeoFips", "TimePeriod"])


# In[11]:


#let dataframe be equal to data imported
full_df = data


# In[12]:


#convert dataframe to float64
full_df = full_df.astype('float64')


# In[ ]:





# In[13]:


# data = pd.read_csv('AbiData.csv')
# full_df = pd.DataFrame({key[:-1]: val.set_index(["GeoFips", "TimePeriod"])["DataValue"] for key, val in data.items()})
# for key, val in full_df.items():
#     try:
#         full_df[key] = val.str.replace(
#             ",","").replace("(NA)",np.NaN).replace("(D)", np.NaN).astype(float)
#     except:
#         continue
            
# full_df


# In[14]:


#get working directory
os.getcwd()

# for a,b,c in os.walk("."):
#     print(a,b,c)


# In[15]:


# full_df = pd.DataFrame({key[:-1]: val.set_index(["GeoFips", "TimePeriod"])["DataValue"] for key, val in GDP_data_dct.items()})
# for key, val in full_df.items():
#     try:
#         full_df[key] = val.str.replace(
#             ",","").replace("(NA)",np.NaN).replace("(D)", np.NaN).astype(float)
#     except:
#         continue
            
# full_df


# In[16]:


full_df["All industry total"]


# In[17]:


# full_df[["Private industries ", "Utilities "]]
list (full_df.keys())


# In[ ]:





# In[18]:


inputs = ['All industry total',"Utilities", "Mining, quarrying, and oil and gas extraction", 'Agriculture, forestry, fishing and hunting']
for key in inputs:
    full_df[key + " 3YMA"] = full_df.reset_index().set_index(["TimePeriod"], drop = False).groupby("GeoFips")[key].rolling(3).mean().shift()
log_df = np.log(full_df)#.groupby("GeoFips").diff()
for key in inputs:
    log_df[key + " Volatility"] = (log_df[key].sub(log_df[key + " 3YMA"])).pow(2).pow(.5)
# for key in inputs:


# - GDP 
# -- layer1

# In[19]:


import json


# In[20]:


toc = """All industry total
1 Private industries
1.1 Agriculture, forestry, fishing and hunting
1.2 Mining, quarrying, and oil and gas extraction
1.3 Utilities
1.4 Construction
1.5 Manufacturing
1.5.1 Durable goods manufacturing
1.5.2 Nondurable goods manufacturing
1.6 Wholesale trade
1.7 Retail trade
1.8 Transportation and warehousing
1.9 Information
1.10 Finance, insurance, real estate, rental, and leasing
1.11 Finance and insurance
1.12 Real estate and rental and leasing
1.13 Professional and business services
1.14 Professional, scientific, and technical services
1.15 Management of companies and enterprises
1.16 Administrative and support and waste management and remediation services
1.17 Educational services, health care, and social assistance
1.18 Educational services
1.19 Health care and social assistance
1.20 Arts, entertainment, recreation, accommodation, and food services
1.21 Arts, entertainment, and recreation
1.22 Accommodation and food services
1.23 Other services (except government and government enterprises)
1.24 Government and government enterprises
1.25 Natural resources and mining
1.26 Trade
1.27 Transportation and utilities
1.28 Manufacturing and information
1.29 Private goods-producing industries 2/
1.30 Private services-providing industries 3/
2 """

myfile = 'GDP'

data = {myfile:{}}
for line in toc.splitlines():
    levels, title = line.split(' ', maxsplit=1)
    levels = levels.rstrip('.').split('.')
    if len(levels) == 1:
        heading = title
        data[myfile][heading] = {}
    elif len(levels) == 2:
        sub_heading = title
        data[myfile][heading][sub_heading] = []
#     if len(levels) == 3:
#         data[myfile][heading][sub_heading].append(title)

print(json.dumps(data, indent=4))


# In[21]:


dct = data["GDP"]
layer1 = list(dct.keys())
layer1.pop()
layer2 = []
for key in layer1:
    layer2 = layer2 + list(dct[key].keys())
layers = {0:layer1,
         1:layer2}
layers


# In[22]:


layer2


# In[23]:


# keys = ['Agriculture, forestry, fishing and hunting',
#  'Mining, quarrying, and oil and gas extraction',
#  'Utilities',
#  'Construction',
#  'Manufacturing',
#  'Wholesale trade',
#  'Retail trade',
#  'Transportation and warehousing',
#  'Information',
#  'Finance, insurance, real estate, rental, and leasing',
#  'Finance and insurance',
#  'Professional and business services',
#  'Educational services, health care, and social assistance',
#  'Arts, entertainment, recreation, accommodation, and food services',
#  'Other services (except government and government enterprises)',
#  'Government and government enterprises']


# In[24]:


# data2 = log_df[layer2].groupby("GeoFips").diff()
# # list(log_df.keys())


# In[25]:


# data2

# list(data2)
# log_df
full_df


# In[26]:


data_dct_ips = {}
data_dct_ips['Log_Data'] = np.log(full_df)
data_dct_ips['Diff1'] = data_dct_ips['Log_Data'].groupby('GeoFips').diff()
data_dct_ips


# In[27]:


data_dct = {}
data_dct["Log Data"] = np.log(full_df).replace([np.inf, -np.inf], np.nan)
data_dct["Diff"] = data_dct["Log Data"].groupby("GeoFips").diff()#.dropna()
data_dct["2Diff"] = data_dct["Diff"].groupby("GeoFips").diff()
data_dct


# write if the variables are sub-components

# In[28]:


# remove duplicate first index column
def reset_index(df):
    name1,name2 = list(df.index.names)[1:]
    ix1, ix2 =df.index.get_level_values(1), df.index.get_level_values(2) 
    df[name1] = ix1
    df[name2] = ix2
    df.reset_index(drop=True, inplace = True)
    df.set_index(["GeoFips", "TimePeriod"], inplace = True)
    
ips_keys = ['Agriculture, forestry, fishing and hunting',
 'Mining, quarrying, and oil and gas extraction',
 'Utilities',
 "All industry total"]
ips_df = data_dct["Diff"][ips_keys]
ips_df = ips_df[ips_df.index.get_level_values("TimePeriod")>2001]
ips_df = ips_df.groupby("GeoFips").apply(lambda x: x.dropna(axis = 1)).dropna()
ips_df


# In[29]:


# ips_df2 = ips_df.groupby("GeoFips").apply(lambda x: x.iloc[x.isnull().values.argmin():])
# reset_index(ips_df2)
# ips_df2 = ips_df2.groupby("GeoFips").apply(lambda x: x.iloc[:x.isnull().values.argmax()])
# reset_index(ips_df2)
# for i in range(3):
#     ips_df2 = ips_df2.groupby("GeoFips").apply(lambda x: x.iloc[x.isnull().values.argmin()+1:])
#     reset_index(ips_df2)
# # .values.argmin()


# In[30]:


# ips_df.iloc[
# data.iloc[:data.A.isnull().values.argmax()]
# ips_df2 = ips_df.groupby("GeoFips").apply(lambda x: x.iloc[x.isnull().values.argmin()+1:])
# ips_df2.groupby("GeoFips").apply(lambda x: x.iloc[:x.isnull().values.argmax()])


# In[31]:


from statsmodels.tsa.stattools import adfuller

def adfuller_table(df):
    df_results = {}
    for key, vector in df.items():
        dftest = adfuller(vector, maxlag = 4, regression = 'c')
        df_results[key] = pd.Series(dftest[0:4], index = ['t-stat', 'p-value', 
                                                         '#Lags Used', 'Number of Observations Used'])
    return pd.DataFrame(df_results).round(2)


# In[32]:


# adfuller_table(data_dct['Diff'])


# In[33]:


import statistics
import math
data = data_dct_ips['Diff1'].replace([np.inf, -np.inf]).fillna(0)
data
data_ips = {}
# data_ips = data.reset_index()
data_ips['Diff']= data
data_ips


# In[34]:


# data_ips_var = ['GeoFips', 'TimePeriod', 'All industry total']
# data_ips_t = data_ips[data_ips_var]
# data_ips_t
# data_ips_t['All industry total'].replace(to_replace = 0, value = 1, inplace=True)


# In[35]:


# data_test = pd.read_csv('ips.csv')
# data_test


# # IPS Test

# In[36]:


def ips_test(data, Firm, Time):
    df = data.set_index([Firm, Time])
    df2 = data.set_index([Time, Firm])
    firms = list(data[Firm].unique())
    times = list(data[Time].unique())
    data = data.set_index(Firm)
    N = len(firms)
    dict_temp = {}

    for firm in firms:
        lag_val = []
        val_diff = []
        lag_val_diff = []
        for i in data[Time].loc[firm]:
            cur_val = i
            if lag_val == []:
                pass
            else:
                val_diff = cur_val - lag_val
            if lag_val_diff == []:
                pass
            else:
                if val_diff != lag_val_diff:
                    raise ValueError("The data does not have constant time variation")
            lag_val = i
            lag_val_diff = val_diff

        for key in df:
            total_temp_stat = 0
            for firm in firms:
                temp_df = data[[key]].loc[firm]
                temp_stat = adfuller(temp_df, maxlag = 0, regression = 'ct')[0]
                total_temp_stat += temp_stat
                t_bar = (1/N)*total_temp_stat
                dict_temp[key] = t_bar

    df_temp = pd.DataFrame([dict_temp]).T
    df_temp.columns = ['Test Stat']
    return df_temp


# Download interest rate data (divisia), and average it annually

# In[37]:


divisia = pd.read_excel("https://centerforfinancialstability.org/amfm/Divisia.xlsx", 
                        sheet_name = "Broad", header  =1, index_col = "Date")

index = divisia.index
keys = divisia.keys()
divisia = divisia.resample("A").mean().rename(columns={keys[0]: "M4",
                                                       keys[2]:"M4 Interest Rate"})
year_lst = [i for i in range(1966, 2023)]
divisia = divisia[["M4", "M4 Interest Rate"]].reset_index()
divisia["TimePeriod"] = year_lst
divisia = divisia.set_index("TimePeriod").drop(columns = "Date")
divisia.head()


# In[38]:


# data["Level"]["$r_{M4}$"] = divisia["M4 Interest Rate"]


# Add/join the interest rates to the original dataframe

# In[39]:


data_df = full_df.copy().reset_index().set_index("TimePeriod")
data_df = data_df.join(divisia).reset_index()     
data_df = data_df.set_index(["GeoFips", "TimePeriod"])    
data_df = data_df.sort_index()
data_df 


# Convert variables to rates and differenced data

# In[40]:


data_dct1 = {}
data_dct1["Log Data"] = np.log(data_df).replace([np.inf, -np.inf], np.nan)
data_dct1["Diff"] = data_dct1["Log Data"].groupby("GeoFips").diff()#.dropna()
#data_dct1["2Diff"] = data_dct1["Diff"].groupby("GeoFips").diff()
data_dct1


# Remove years with nan

# In[41]:


# remove duplicate first index column
def reset_index(df):
    name1,name2 = list(df.index.names)[1:]
    ix1, ix2 =df.index.get_level_values(1), df.index.get_level_values(2) 
    df[name1] = ix1
    df[name2] = ix2
    df.reset_index(drop=True, inplace = True)
    df.set_index(["GeoFips", "TimePeriod"], inplace = True)
    
ips_keys = ['Agriculture, forestry, fishing and hunting',
 'Mining, quarrying, and oil and gas extraction',
 'Utilities',
 "All industry total",
           "M4",
           "M4 Interest Rate"]
ips_df1 = data_dct1["Log Data"][ips_keys]
ips_df1 = ips_df1[ips_df1.index.get_level_values("TimePeriod")>2001]
ips_df1 = ips_df1.groupby("GeoFips").apply(lambda x: x.dropna(axis = 1)).dropna()

ips_df1["M4 Interest Rate"] = data_df["M4 Interest Rate"]
ips_df1


# In[42]:


# years = ips_df.groupby("TimePeriod").mean().index
# entities = ips_df.groupby("GeoFips").mean().index
# years, entities


# enter data that is the same for *every* year entry
# 
# using year as example for data entry here

# In[43]:


# df = ips_df.copy()
# df["New"] = np.nan
# for year in years:
#     df["New"][df.index.get_level_values("TimePeriod") == year] = year
# df


# In[44]:


ips_dct = {"Log":ips_df}
ips_dct["Rates"] = ips_dct["Log"].groupby("GeoFips").diff()
ips_dct["Diff"] = ips_dct["Rates"].groupby("GeoFips").diff()


# Run the IPS test

# In[45]:


# import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller

def ips_test(data):
    index_name, sub_index_name = data.index.names
    index = list(data.reset_index()[index_name].unique())
    N = len(index)
    t_stats = {}
    df_index_dict = {}
    for ix in index:
        slice_df = data.loc[ix]
#         print(slice_df)
        t_stats[ix] = {}
        for key in slice_df.keys():
            try:
                t_stat = adfuller(slice_df[key], maxlag = 1, regression = 'c')[0]
                t_stats[ix][key] = t_stat
            except:
                print("Error:", key)
    t_stats = pd.DataFrame(t_stats).T

    return t_stats.mean()
   
ips_results = {}
for key, val in ips_dct.items():
    ips_results[key] = ips_test(val.dropna())
pd.DataFrame(ips_results).dropna()


# In[46]:


ips_dct2 = {"Log":ips_df1}
ips_dct2["Rates"] = ips_dct2["Log"].groupby("GeoFips").diff()
ips_dct2["Diff"] = ips_dct2["Rates"].groupby("GeoFips").diff()


# In[47]:


# import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller

def ips_test(data):
    index_name, sub_index_name = data.index.names
    index = list(data.reset_index()[index_name].unique())
    N = len(index)
    t_stats = {}
    df_index_dict = {}
    for ix in index:
        slice_df = data.loc[ix]
#         print(slice_df)
        t_stats[ix] = {}
        for key in slice_df.keys():
            try:
                t_stat = adfuller(slice_df[key], maxlag = 1, regression = 'c')[0]
                t_stats[ix][key] = t_stat
            except:
                print("Error:", key)
    t_stats = pd.DataFrame(t_stats).T

    return t_stats.mean()
   
ips_results = {}
for key, val in ips_dct2.items():
    ips_results[key] = ips_test(val.dropna())
pd.DataFrame(ips_results).dropna()


# In[48]:


import pingouin

plt.rcParams.update({'font.size': 30})
plt.rcParams['axes.xmargin'] = .001
plt.rcParams['axes.ymargin'] = .005
def full_corr_plot(data, color = "C0", pcorr = False):
    if pcorr == True:
        corr_df = data.pcorr()
    elif pcorr == False:
        corr_df = data.corr()
    keys = list(corr_df.keys())
    dim = len(keys)

    fig, ax = plt.subplots(figsize = (30, 30))
    a = pd.plotting.scatter_matrix(data, c = color, 
                                   s = 200, alpha = .1, ax=ax)  
    for i in range(len(keys)):
        x = keys[i]
        for j in range(len(keys)):
            y = keys[j]
            a[i][j].set_xticklabels([])
            a[i][j].set_yticklabels([])
            a[i][j].set_title("$\\rho :" + str(corr_df.round(2)[x][y])+ "$", y = .88, x = 0.01, ha = "left")        
    plt.suptitle("Correlation\n(Color: y)",y = .96, fontsize = 80)
plot_df = ips_dct2['Diff'].dropna()
plot_df.rename(columns = {key:key.replace(" ", "\n") for key in plot_df.keys()}, inplace = True)
plot_keys = list(plot_df.keys())
full_corr_plot(plot_df, color = plot_df[plot_keys[0]], pcorr = True)
# y_var = ['Agriculture, forestry, fishing and hunting']
# x_vars = ['Mining, quarrying, and oil and gas extraction', 'Utilities', 'Construction', 'Manufacturing']
# corr_var = y_var + x_vars
# corr_data = log_df[corr_var]
# corr_data.corr().round(3)


# In[49]:


# . . .
def corr_matrix_heatmap(data, pp = False):  
    #Create a figure to visualize a corr matrix  
    fig, ax = plt.subplots(figsize=(20,20))  
    # use ax.imshow() to create a heatmap of correlation values  
    # seismic mapping shows negative values as blue and positive values as red  
    im = ax.imshow(data, norm = plt.cm.colors.Normalize(-1,1), cmap = "seismic")  
    # create a list of labels, stacking each word in a label by replacing " "  
    # with "\n"  
    labels = data.keys()  
    num_vars = len(labels)  
    tick_labels = [lab.replace(" ", "\n") for lab in labels]  
    # adjust font size according to the number of variables visualized  
    tick_font_size = 120 / num_vars  
    val_font_size = 200 / num_vars  
    plt.rcParams.update({'font.size': tick_font_size}) 
    # prepare space for label of each column  
    x_ticks = np.arange(num_vars)  
    # select labels and rotate them 90 degrees so that they are vertical  
    plt.xticks(x_ticks, tick_labels, fontsize = tick_font_size, rotation = 90)  
    # prepare space for label of each row  
    y_ticks = np.arange(len(labels))  
    # select labels  
    plt.yticks(y_ticks, tick_labels, fontsize = tick_font_size)  
    # show values in each tile of the heatmap  
    for i in range(len(labels)):  
        for j in range(len(labels)):  
            text = ax.text(i, j, str(round(data.values[i][j],2)),  
                           fontsize= val_font_size, ha="center",   
                           va="center", color = "w")  
    #Create title with Times New Roman Font  
    title_font = {"fontname":"Times New Roman"}  
    plt.title("Correlation", fontsize = 50, **title_font)  
    #Call scale to show value of colors 
    cbar = fig.colorbar(im)
    plt.show()
    if pp != False:
        pp.savefig(fig, bbox_inches="tight")
    plt.close()

#. . . 
# . . .
corr_matrix_heatmap(plot_df.corr())


# In[50]:


# list(plot_df)
# ips_df
plot_df.rename(columns = {key:key[:4].replace("\n", "") for key in plot_df.keys()}, inplace = True)
list(plot_df.keys())


# In[51]:


from matplotlib.patches import ArrowStyle
import copy
from matplotlib.backends.backend_pdf import PdfPages

undirected_graph = {key:[] for key in plot_df.keys()}
for x in undirected_graph:
    remaining_vars = [y for y in plot_df.keys() if y != x]
    for y in remaining_vars:
        undirected_graph[x].append(y)

p_value = .01
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import PC
c = PC(plot_df)
max_cond_vars = len(plot_df.keys()) - 2

model = c.estimate(return_type = 'pdag', variant= 'parallel', significance_level = p_value,
                  max_cond_vars = max_cond_vars, ci_test = 'pearsonr')
edges = model.edges

pp = PdfPages("DAGOutputs1.pdf")

def graph_DAG(edges, df, title = ""):
    graph = nx.Graph()
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
        pcorr = df[[edge[0], edge[1]]+keep_controls].pcorr()
        edge_labels[edge] = str(round(pcorr[edge[0]].loc[edge[1]],2))
    graph.add_edges_from(edges)
    color_map = ['C0' for g in graph]
    
    fig, ax = plt.subplots(figsize = (20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize = 30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                    arrows=True, font_color ='k', font_size=26, alpha=1, width = 1,
                    edge_color = 'C1',
                     arrowstyle=ArrowStyle('Fancy, head_length=3, head_width=1.5, tail_width=.1'), ax = ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches = "tight")

graph_DAG(edges, plot_df, title = 'Directed Acyclic Graph')


pp.close()                                                            
edges


# In[52]:


plot_df


# In[53]:


# !pip install geopandas


# In[ ]:





# In[54]:


# counties = gpd.read_file("C:/Users/abiodun.idowu/OneDrive - North Dakota University System/Desktop/PhD/BEA project/notebook_to_start/tl_2022_us_county.shp")


# In[55]:


# print(counties.head())

# # Rename the county name column to 'county_name'
# counties = counties.rename(columns={'NAME': 'county_name'})

# # Create a GeoDataFrame with just the county name and geometry columns
# counties_gdf = counties[['county_name', 'geometry']]


# county_gdf = counties.rename(columns={'GEOID': 'GeoFips'})

# # Check the new structure of the data

# # print(counties_gdf.head())

# # Merge the geometry column from the GeoDataFrame with the 'plot_df' DataFrame
# # merged_df = plot_df.merge(county_gdf[['GeoFips', 'geometry']], on='GeoFips')

# # 'merged_df' now contains both the attribute data from 'plot_df' and th


# In[56]:


# county_gdf['GeoFips'] = county_gdf['GeoFips'].astype(str)


# In[57]:


# merge_df = pd.concat([plot_df, county_gdf]).groupby('GeoFips')


# In[58]:


# merge_df


# In[59]:


# county_gdf


# In[60]:


# print(plot_df.dtypes)
# print(county_gdf.dtypes)


# In[61]:


# county_gdf['GeoFips'] = county_gdf['GeoFips'].astype(int)


# In[62]:


# merge_df = pd.merge(plot_df.reset_index(), county_gdf, left_on='GeoFips', right_on='GeoFips')


# In[63]:


# merge_df = merge_df.set_index('GeoFips')


# In[64]:


# gdf = merge_df


# In[65]:


# gdf = gdf.dropna(subset=['geometry'])


# In[66]:


# gdf


# In[ ]:





# In[67]:


# import geopandas as gpd
# import pandas as pd


# In[68]:


# gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

# # Check if a geometry intersects another geometry in the GeoDataFrame
# intersects = gdf.iloc[0].geometry.intersects(gdf.iloc[1].geometry)
# print(intersects)


# In[69]:


# for index, GeoFips in gdf.iterrows():
#     neighbors = gdf[~gdf.geometry.disjoint(GeoFips.geometry)].county_name.tolist()
#     neighbor = [name for name in neighbors if GeoFips.county_name != name]
#     gdf.at[index, "NEIGHBORS"] = ", ".join(neighbors)


# In[ ]:





# In[ ]:





# In[70]:


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
fips_name = "fips_code"
map_data = import_geo_data(
    filename = "countiesWithStatesAndPopulation.shp",
    index_col = "Date", FIPS_name= fips_name)
map_data


# In[71]:


map_data


# In[72]:


map_data.to_csv('map_data.csv', index=True)


# In[73]:


map_data.dtypes


# In[74]:


# for index, fips_code in map_data.iterrows():
#     neighbors = map_data[~map_data.geometry.disjoint(fips_code.geometry)].index.tolist()
#     neighbors = [name for name in neighbors if name not in fips_code.index]
#     print(neighbors)
# map_data.at[index, "NEIGHBORS"] = neighbors


# In[75]:


# for index, fips_code in map_data.iterrows():
#     neighbors = map_data[~map_data.geometry.disjoint(fips_code.geometry)].index.tolist()
#     neighbors = [name for name in neighbors if name != index]
#     print(index, neighbors)
# map_data.at[index, "NEIGHBORS"] = neighbors


# In[76]:


# for index, fips_code in map_data.iterrows():
#     neighbors = map_data[~map_data.geometry.disjoint(fips_code.geometry)].index.tolist()
#     neighbors = [int(name) for name in neighbors if name != index]
#     print(index, neighbors)
# map_data.at[index, "NEIGHBORS"] = neighbors


# In[ ]:





# In[77]:


# find_neighbors(year_data)


# In[ ]:





# In[78]:


# for year in range (2004, 2020):
#     year_data = full_df.loc[year]
#     year_data = map_data.join(year_data).dropna(subset = ["All industry total"])
#     find_neighbors(year_data)    
# #     year_data = year_data.join(map_data["NEIGHBORS"])
# #     year_data["NeighborGDP"]
# #     print( year_data["All industry total"].loc[year_data.loc[1001]["NEIGHBORS"]].sum())
# #     full_df.loc[year, "All industry total"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
# #                           axis = 1)
#     full_df.loc[year, "All industry total"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
#                           axis = 1)


# In[79]:


def find_neighbors(df):
    for index, fips_code in df.iterrows():

        neighbors = df[~df.geometry.disjoint(fips_code.geometry)].index.tolist()
#         neighbors = [int(name) for name in neighbors if name != index]
        print(index, neighbors)
        df.at[index, "NEIGHBORS"] = neighbors
map_data["NEIGHBORS"] = ""
find_neighbors(map_data)


# In[ ]:





# In[80]:


full_df = full_df.reset_index()
full_df["FIPS"] = full_df["GeoFips"]
full_df = full_df.set_index(["TimePeriod","GeoFips"])
full_df.dropna(subset = ["All industry total"], inplace = True)


# In[81]:


# full_df["NeighborGDP"] = np.NaN
# year_df_dict =[] 
# for year in full_df.index.get_level_values("TimePeriod").unique():# range (2001, 2020):
#     year_data = full_df.loc[year]
#     year_data = map_data.join(year_data).dropna(subset = ["All industry total"])
#     find_neighbors(year_data)    
# #     year_data = year_data.join(map_data["NEIGHBORS"])
# #     year_data["NeighborGDP"]
# #     print( year_data["All industry total"].loc[year_data.loc[1001]["NEIGHBORS"]].sum())
# #     full_df.loc[year, "All industry total"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
# #                           axis = 1)
#     full_df["NeighborGDP"].loc[year,year_data.index] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
#                           axis = 1)


# In[ ]:





# In[ ]:





# In[82]:


# # full_df.dropna(subset = ["NeighborGDP"])#.index.get_level_values("TimePeriod").unique()
# full_df.loc[year, "NeighborGDP"].loc[year_data.index] = year_data.apply(
#     lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
#                           axis = 1)
# full_df.loc[year, "NeighborGDP"].loc[year_data.index]
# # full_df.loc[year]
# # full_df.loc[year].loc[year_data.index]


# In[83]:


import datetime
# full_df["NeighborGDP"][full_df.index.get_level_values("GeoFips").isin(year_data["FIPS"])].loc[year]



# In[84]:


try_df = full_df.reset_index()
try_df.set_index(["GeoFips"])


# Create neighbors and sum of their GDP

# In[85]:


years_list = try_df["TimePeriod"].unique()
full_df["NeighborGDP"] = np.NaN
year_df_dict ={}
for year in years_list:    
    year_data = full_df.loc[year]
    year_data = map_data.join(year_data).dropna(subset = ["All industry total"])
    find_neighbors(year_data)    
#     year_data = year_data.join(map_data["NEIGHBORS"])
#     year_data["NeighborGDP"]
#     print( year_data["All industry total"].loc[year_data.loc[1001]["NEIGHBORS"]].sum())
#     full_df.loc[year, "All industry total"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
#                           axis = 1)
    full_df["NeighborGDP"].loc[year,year_data.index] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
                          axis = 1)    
    
    
##############################################################################3    
    
    
    year_data["NeighborGDP"] = year_data.apply(
        lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
                              axis = 1)
    year_data["Year"] = datetime.datetime(year,1,1)
    trial = year_data    
    try_df = full_df.reset_index().set_index(["GeoFips"])
    year_df_dict[year] = pd.merge(try_df.loc[try_df["TimePeriod"]==year], trial.loc[trial["Year"].dt.year==year],
                       left_index=True, right_index=True)


# In[86]:


year_df_dict[2020]


# In[87]:


df_list = (year_df_dict.values())
df_list


# In[88]:


from functools import reduce

df = pd.concat([d for d in df_list], axis=0, join='inner').set_index(["Year", "FIPS_y"])
df


# In[ ]:





# In[89]:


df["GDP_weigh"] = df["NeighborGDP_y"]/df["All industry total_y"]


# In[90]:


df


# In[91]:


df.keys()


# In[92]:


#sum of gdp/self area (use .area)

#sum of gdp/self population

#build out some maps from these variables, correlation stats of your own gdp against these measures (do initial analytics)

#partial correlations betw your gdp and neighbors gdp

#create maps


# In[93]:


df["GDP_area"] = df["NeighborGDP_y"]/df["ALAND"]
df


# In[94]:


df["GDP_area"] = df["NeighborGDP_y"]/df["ALAND"]
df["GDP_pop"] = df["NeighborGDP_y"]/df["Population"]
df["ownGDP_pop"] = df["All industry total_y"]/df["Population"]


# In[95]:


df


# In[96]:


# df.to_csv('path/to/output.csv', index=False)

#df.to_csv('stat712.csv', index=False)


# In[97]:


df1 = df.copy().reset_index().set_index("TimePeriod")
df1 = df1.join(divisia).reset_index()     
df1 = df1.set_index(["FIPS_y", "TimePeriod"])    
df1 = df1.sort_index()
df1
df2_key = ['Agriculture, forestry, fishing and hunting_y',
 'Mining, quarrying, and oil and gas extraction_y',
 'Utilities_y',
 "All industry total_y", 'NeighborGDP_y', 'GDP_weigh', 'M4', 'M4 Interest Rate']
df2 = df1[df2_key]


# In[ ]:





# In[98]:


df5_key = ['All industry total_y',
 'GDP_area',
 'GDP_pop', 'NeighborGDP_y']
df6 = df[df5_key]


# In[ ]:





# In[ ]:





# In[99]:


df3 = {}
df3["Log data"] = np.log(df2).replace([np.inf, -np.inf], np.nan)
df3["Diff"] = df3["Log data"].groupby("FIPS_y").diff().dropna()
df3["Diff2"] = df3["Diff"].groupby("FIPS_y").diff() 
df3


# ## Correlation

# In[100]:


plt.rcParams.update({'font.size': 30})
plt.rcParams['axes.xmargin'] = .001
plt.rcParams['axes.ymargin'] = .005
df7 = df6.dropna()
df7.rename(columns = {key:key.replace(" ", "\n") for key in df7.keys()}, inplace = True)
df7_keys = list(df7.keys())
full_corr_plot(df7, color = df7[df7_keys[0]], pcorr = True)


# In[101]:


corr_matrix_heatmap(df7.corr())


# In[102]:


df8 = {}
df8["Log"] = np.log(df6)
df8["Diff"] = df8["Log"].groupby("FIPS_y").diff()
df8["Diff2"] = df8["Diff"].groupby("FIPS_y").diff()


# In[ ]:





# In[103]:


plt.rcParams.update({'font.size': 30})
plt.rcParams['axes.xmargin'] = .001
plt.rcParams['axes.ymargin'] = .005
df9 = df8["Diff"].dropna()
df7.rename(columns = {key:key.replace(" ", "\n") for key in df7.keys()}, inplace = True)
df9_keys = list(df9.keys())
full_corr_plot(df9, color = df9[df9_keys[0]], pcorr = True)


# In[104]:


corr_matrix_heatmap(df9.corr())


# In[105]:


type(map_data)


# In[106]:


corrdf = map_data.copy()


# In[107]:


df1.reset_index(inplace=True)


# In[108]:


df_test = df1[(df1['TimePeriod'] >= 2001) & (df1['TimePeriod'] <= 2020)]


# In[109]:


correlation = df_test.groupby('FIPS_y')['All industry total_y', 'NeighborGDP_y'].corr().iloc[0::2]['NeighborGDP_y']


# In[110]:


import statsmodels.api as sm


# In[111]:


df_reg = df_test.groupby('FIPS_y').apply(lambda x: sm.OLS(x['All industry total_y'], sm.add_constant(x['NeighborGDP_y'])).fit().params['NeighborGDP_y'])


df_reg2 = df_test.groupby('FIPS_y').apply(lambda x: sm.OLS(x['All industry total_y'], sm.add_constant(x['NeighborGDP_y'])).fit().params['NeighborGDP_y']).reset_index(name='Regression_Coefficient')


# In[112]:


df_test1 = df_test.copy()

df_test1 = df_test1.set_index(["FIPS_y", "TimePeriod"])

df_test_key = ['All industry total_y', 'NeighborGDP_y']

df_test2 = df_test1[df_test_key]


# In[ ]:





# In[113]:


df_test_reg = {}
df_test_reg["Log"] = np.log(df_test2)
df_test_reg["Diff"] = df_test_reg["Log"].groupby("FIPS_y").diff().dropna()


# In[114]:


df_reg2 = df_test_reg["Diff"].groupby('FIPS_y').apply(lambda x: sm.OLS(x['All industry total_y'], sm.add_constant(x['NeighborGDP_y'])).fit().params['NeighborGDP_y']).reset_index(name='Regression_Coefficient')


# In[115]:


df_reg2.dropna()


# In[116]:


# Plot of the distribution for regression coefficients
plot = plt.hist(df_reg2['Regression_Coefficient'], bins=20, edgecolor='black')
plt.axvline(np.mean(df_reg2['Regression_Coefficient']), color='red', linestyle='dashed', linewidth=2)
plt.xlabel('Regression Coefficient')
plt.ylabel('Frequency')
plt.title('Distribution of Regression Coefficients')


min_value = np.min(df_reg2['Regression_Coefficient'])
max_value = np.max(df_reg2['Regression_Coefficient'])
plt.xlim(min_value, max_value)


plt.savefig('plot.png')


# In[117]:


os.getcwd()


# In[118]:


# Histogram of the regression coefficients
import seaborn as sns
sns.displot(data=df_reg2, x='Regression_Coefficient', kde=True, color='blue')
plt.xlabel('Regression Coefficient')
plt.ylabel('Frequency')
plt.title('Distribution of Regression Coefficients')


# In[119]:


df8 = {}
df8["Log"] = np.log(df6)
df8["Diff"] = df8["Log"].groupby("FIPS_y").diff()


# In[120]:


#merge regression coefficients with map data and plot

df_reg2 = df_reg2.rename(columns={'FIPS_y':'fips_code'})

df_reg2['fips_code'] = df_reg2['fips_code'].astype(int)


# In[121]:


df_reg2 = df_reg2.set_index('fips_code')


# In[ ]:





# In[122]:


map_data.reset_index().set_index('fips_code', inplace=True)


# In[123]:


df_reg3 = pd.merge(map_data, df_reg2, left_index=True, right_on='fips_code')


# In[124]:


df_reg3.columns.tolist()


# In[125]:


# df_reg3 = df_reg3.rename(columns={'fips_code':'index', 'index': 'col'})



# In[ ]:





# In[126]:


# !pip install plotly


# In[127]:


import plotly.express as px

fig = px.choropleth(df_reg3, geojson=df_reg3.geometry.__geo_interface__, locations=df_reg3.index, color='Regression_Coefficient')
fig.update_geos(fitbounds='locations', visible=False)
fig.show()


# In[128]:


fig.write_html('fig.html')


# In[129]:


df_reg2.to_csv('stat_712.csv', index=False)


# In[130]:


df_corr = {}
df_corr['corr_gdp'] = correlation
df_corr['corr_gdp_pop'] = df_test.groupby('FIPS_y')['GDP_pop', 'ownGDP_pop'].corr().iloc[0::2]['ownGDP_pop']


# In[131]:


corr = pd.DataFrame(correlation)

corr_pop = pd.DataFrame(df_corr['corr_gdp_pop'])


# In[ ]:





# In[132]:


full_df.keys()


# In[ ]:





# Sum of neighbor energy sector

# In[133]:


years_list = try_df["TimePeriod"].unique()
full_df["Neighborenergy"] = np.NaN
year_df_dict ={} 
for year in years_list:    
    year_data = full_df.loc[year]
    year_data = map_data.join(year_data).dropna(subset = ["Mining, quarrying, and oil and gas extraction"])
    find_neighbors(year_data)    
#     year_data = year_data.join(map_data["NEIGHBORS"])
#     year_data["NeighborGDP"]
#     print( year_data["All industry total"].loc[year_data.loc[1001]["NEIGHBORS"]].sum())
#     full_df.loc[year, "All industry total"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
#                           axis = 1)
    full_df["Neighborenergy"].loc[year,year_data.index] = year_data.apply(lambda row: year_data["Mining, quarrying, and oil and gas extraction"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
                          axis = 1)    
    
    
##############################################################################3    
    
    
    year_data["Neighborenergy"] = year_data.apply(
        lambda row: year_data["Mining, quarrying, and oil and gas extraction"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
                              axis = 1)
    year_data["Year"] = datetime.datetime(year,1,1)
    trial = year_data    
    try_df = full_df.reset_index().set_index(["GeoFips"])
    year_df_dict[year] = pd.merge(try_df.loc[try_df["TimePeriod"]==year], trial.loc[trial["Year"].dt.year==year],
                       left_index=True, right_index=True)


# In[134]:


df_list_energy = (year_df_dict.values())


# In[135]:


df_energy = pd.concat([d for d in df_list_energy], axis=0, join='inner').set_index(["Year", "FIPS_y"])
df_energy


# In[136]:


df_energy = df_energy.merge(df['NeighborGDP_y'], left_index=True, right_index=True)


# In[137]:


#df_energy

df_energy = df_energy.rename(columns={'Neighborenergy_y': 'Neighborenergy', 'NeighborGDP_y_y':'NeighborGDP'})

df_energy = df_energy.drop('NeighborGDP_y_x', axis=1)


# In[138]:


gdp_energy = df_energy.copy()
gdp_energy_key = ['All industry total_y','Mining, quarrying, and oil and gas extraction_y', 'NeighborGDP', 'Neighborenergy']
gdp_energy1 = gdp_energy[gdp_energy_key]
gdp_energy1 = gdp_energy1.rename(columns={'Mining, quarrying, and oil and gas extraction_y' : 'energy'})


# In[139]:


#Create a dictionary for logged data and rates data
energy_df = {}
energy_df["Log"] = np.log(gdp_energy1).replace([np.inf, -np.inf], np.nan)
energy_df["Diff"] = energy_df["Log"].groupby("FIPS_y").diff().dropna()


# In[140]:


energy_df["Diff"]


# In[141]:


#Create a dictionary of all the regression coefficients for various combinations
reg_coeff ={}

reg_coeff["gdp_neighborgdp_coeff"] = energy_df["Diff"].groupby('FIPS_y').apply(lambda x: sm.OLS(x['All industry total_y'], sm.add_constant(x['NeighborGDP'])).fit().params['NeighborGDP']).reset_index(name='gdp_neighborgdp_coeff').set_index('FIPS_y')

reg_coeff["gdp_neighborenergy_coeff"] = energy_df["Diff"].groupby('FIPS_y').apply(lambda x: sm.OLS(x['All industry total_y'], sm.add_constant(x['Neighborenergy'])).fit().params['Neighborenergy']).reset_index(name='gdp_neighborenergy_coeff').set_index('FIPS_y')

reg_coeff["energy_neighborenergy_coeff"] = energy_df["Diff"].groupby('FIPS_y').apply(lambda x: sm.OLS(x['energy'], sm.add_constant(x['Neighborenergy'])).fit().params['Neighborenergy']).reset_index(name='energy_neighborenergy_coeff').set_index('FIPS_y')

#df_reg2 = df_test_reg["Diff"].groupby('FIPS_y').apply(lambda x: sm.OLS(x['All industry total_y'], sm.add_constant(x['NeighborGDP_y'])).fit().params['NeighborGDP_y']).reset_index(name='Regression_Coefficient')


# In[142]:


reg_coeff["gdp_neighborgdp_coeff"]


# In[143]:


#merge all the regression coefficients to one dataframe

reg_coeff_map = pd.merge(reg_coeff["gdp_neighborgdp_coeff"] , reg_coeff["gdp_neighborenergy_coeff"], 
                         left_index=True, right_on='FIPS_y')

reg_coeff_map = pd.merge(reg_coeff_map, reg_coeff["energy_neighborenergy_coeff"], 
                         left_index=True, right_on='FIPS_y')

# reg_coeff_map = reg_coeff_map.rename(columns={'gdp_neighborenergy_coeff_x':'gdp_neighborenergy_coeff',
#                                              'gdp_neighborenergy_coeff_y':'gdp_neighborenergy_coeff'})




# Energy vs Neighbor energy sector

# In[144]:


#merge data with location dataframe
reg_energy = reg_coeff["energy_neighborenergy_coeff"]
reg_energy = reg_energy.reset_index()
reg_energy = reg_energy.rename(columns={'FIPS_y':'fips_code'})
reg_energy = reg_energy.set_index('fips_code')
map_data.reset_index().set_index('fips_code', inplace=True)
map_energy = pd.merge(map_data, reg_energy, left_index=True, right_on='fips_code')

#Plot the dynamic map
# Get the maximum absolute value in the DataFrame
# Filter out rows for Alaska and Hawaii using their FIPS codes 
map_energy = map_energy[~map_energy['state'].isin(['Alaska', 'Hawaii'])]

max_value = max(abs(map_energy['energy_neighborenergy_coeff'].max()), abs(map_energy['energy_neighborenergy_coeff'].min()))

# Create the choropleth map
fig_energy = px.choropleth(map_energy, geojson=map_energy['geometry'], locations=map_energy.index, color='energy_neighborenergy_coeff',
                    color_continuous_scale='RdBu', color_continuous_midpoint=0, range_color=[-max_value, max_value])
fig_energy.update_geos(fitbounds='locations', resolution=110, scope="usa",
    showsubunits=True, subunitcolor="Black", subunitwidth=0.5)
fig_energy.update_traces(marker_line_width=0.5)
fig_energy.show()

fig_energy.write_html('fig_energy.html')


# In[145]:


# Create the choropleth map with a specific color range
fig_energy1 = px.choropleth(map_energy, geojson=map_energy['geometry'], locations=map_energy.index, color='energy_neighborenergy_coeff',
                    color_continuous_scale='RdBu', color_continuous_midpoint=0, range_color=[-5, 5])
fig_energy1.update_geos(fitbounds='locations', resolution=110, scope="usa",
    showsubunits=True, subunitcolor="Black", subunitwidth=0.5)
fig_energy1.update_traces(marker_line_width=0.5)

fig_energy1.show()

fig_energy1.write_html('fig_energy1.html')


# GDP vs neighbor energy

# In[146]:


#merge data with location data
reg_gdpenergy = reg_coeff["gdp_neighborenergy_coeff"]
reg_gdpenergy = reg_gdpenergy.reset_index()
reg_gdpenergy = reg_gdpenergy.rename(columns={'FIPS_y':'fips_code'})
reg_gdpenergy = reg_gdpenergy.set_index('fips_code')
map_data.reset_index().set_index('fips_code', inplace=True)
map_gdpenergy = pd.merge(map_data, reg_gdpenergy, left_index=True, right_on='fips_code')


#Plot the map
# Get the maximum absolute value in the DataFrame
map_gdpenergy = map_gdpenergy[~map_gdpenergy['state'].isin(['Alaska', 'Hawaii'])]
max_value = max(abs(map_gdpenergy['gdp_neighborenergy_coeff'].max()), abs(map_gdpenergy['gdp_neighborenergy_coeff'].min()))

# Create the choropleth map
fig_gdp_e = px.choropleth(map_gdpenergy, geojson=map_gdpenergy['geometry'], locations=map_gdpenergy.index, color='gdp_neighborenergy_coeff',
                    color_continuous_scale='RdBu', color_continuous_midpoint=0, range_color=[-5, 5]) #range_color=[-max_value, max_value])
fig_gdp_e.update_geos(fitbounds='locations', resolution=110, scope="usa",
    showsubunits=True, subunitcolor="Black", subunitwidth=0.5)
fig_gdp_e.update_traces(marker_line_width=0.5)
fig_gdp_e.show()

fig_gdp_e.write_html('fig_gdpVSenergy.html')


# GDP vs Neighbor GDP

# In[147]:


#merge data with location data
reg_gdpgdp = reg_coeff["gdp_neighborgdp_coeff"]
reg_gdpgdp = reg_gdpgdp.reset_index()
reg_gdpgdp = reg_gdpgdp.rename(columns={'FIPS_y':'fips_code'})
reg_gdpgdp = reg_gdpgdp.set_index('fips_code')
map_data.reset_index().set_index('fips_code', inplace=True)
map_gdpgdp = pd.merge(map_data, reg_gdpgdp, left_index=True, right_on='fips_code')


#Plot dynamic map
# Get the maximum absolute value in the DataFrame
map_gdpgdp = map_gdpgdp[~map_gdpgdp['state'].isin(['Alaska', 'Hawaii'])]
max_value = max(abs(map_gdpgdp["gdp_neighborgdp_coeff"].max()), abs(map_gdpgdp["gdp_neighborgdp_coeff"].min()))

# Create the choropleth map
fig_gdp = px.choropleth(map_gdpgdp, geojson=map_gdpgdp['geometry'], locations=map_gdpgdp.index, 
                    color='gdp_neighborgdp_coeff', color_continuous_scale='RdBu', 
                    color_continuous_midpoint=0, range_color=[-max_value, max_value])
fig_gdp.update_geos(visible=False, resolution=110, scope="usa",
    showsubunits=True, subunitcolor="Black", subunitwidth=0.5)
fig_gdp.update_geos(fitbounds='locations')
fig_gdp.update_traces(marker_line_width=0.5)
fig_gdp.show()

fig_gdp.write_html('fig_gdpVSgdp.html')


# In[148]:


# dfmp = map_gdpenergy


# In[149]:


# dfmp = dfmp.reset_index()


# In[150]:


# north_dakota = dfmp[dfmp['state'] == 'North Dakota']#['fips_code']
# north_dakota.to_csv('stat_713.csv')


# In[151]:


dfaisus_energy = map_energy

dfaisus_gdpenergy = reg_gdpenergy

dfaisus_gdpgdp = reg_gdpgdp


# In[ ]:





# In[152]:


dfaisus1 = pd.merge(dfaisus_energy, dfaisus_gdpenergy, left_index=True, right_on='fips_code')


# In[153]:


dfaisus1 = pd.merge(dfaisus1, dfaisus_gdpgdp, left_index=True, right_on='fips_code')


# In[154]:


dfaisus1 = dfaisus1.reset_index()


# In[ ]:





# In[155]:


# dfaisus_energy=dfaisus_energy.reset_index()


# In[156]:


# dfaisus_energy = map_energy.reset_index()

# map_gdpenergy


# In[157]:


# Calculate whether the coefficients are positive or negative
dfaisus1['sign_energy'] = np.where(dfaisus1['energy_neighborenergy_coeff'] > 0, 'Positive', 'Negative')

# Count the occurrences of positive and negative coefficients
coeff_counts = dfaisus1['sign_energy'].value_counts()

# Create a pie chart
plt.figure(figsize=(12, 12))
colors = ['green', 'red']
plt.pie(coeff_counts, labels=coeff_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Coefficients')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[158]:


# Separate positive and negative coefficients
positive_coeffsenergy = dfaisus1[dfaisus1['energy_neighborenergy_coeff'] > 0]
negative_coeffsenergy = dfaisus1[dfaisus1['energy_neighborenergy_coeff'] < 0]

# Calculate the range of coefficients within some standard deviations
num_std_devs = 2  # Change this value as needed

# Create scatter plots
plt.figure(figsize=(10, 6))
plt.scatter(positive_coeffsenergy['fips_code'], positive_coeffsenergy['energy_neighborenergy_coeff'], 
            color='green', label='Positive Coefficients')
plt.scatter(negative_coeffsenergy['fips_code'], negative_coeffsenergy['energy_neighborenergy_coeff'], 
            color='red', label='Negative Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot of Positive and Negative Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot positive coefficients
plt.figure(figsize=(10, 6))
plt.scatter(positive_coeffsenergy['fips_code'], positive_coeffsenergy['energy_neighborenergy_coeff'], 
            color='green', label='Positive Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Positive Regression Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot negative coefficients
plt.figure(figsize=(10, 6))
plt.scatter(negative_coeffsenergy['fips_code'], negative_coeffsenergy['energy_neighborenergy_coeff'], 
            color='red', label='Negative Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Negative Regression Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# Plot the choropleth map of counties within positive coefficients
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
positive_coeffsenergy.plot(ax=ax, edgecolor='black', cmap='coolwarm')
ax.set_title('Counties with positive regression coefficients'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# Plot the choropleth map of counties within positive coefficients
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
negative_coeffsenergy.plot(ax=ax, edgecolor='black', cmap='coolwarm')
ax.set_title('Counties with negative regression coefficients'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# In[159]:


# Calculate mean and standard deviation of coefficients
coeff_meanenergy = dfaisus1['energy_neighborenergy_coeff'].mean()
coeff_stdenergy = dfaisus1['energy_neighborenergy_coeff'].std()

# Calculate the range of coefficients within some standard deviations
num_std_devs = 2  # Change this value as needed
lower_bound = coeff_meanenergy - num_std_devs * coeff_stdenergy
upper_bound = coeff_meanenergy + num_std_devs * coeff_stdenergy

# Filter coefficients within the specified range
# Filter coefficients within and outside the specified range
coefficients_within_range1 = dfaisus1[
    (dfaisus1['energy_neighborenergy_coeff'] >= lower_bound) & (dfaisus1['energy_neighborenergy_coeff'] <= upper_bound)
]
coefficients_outside_range1 = dfaisus1[
    (dfaisus1['energy_neighborenergy_coeff'] < lower_bound) | (dfaisus1['energy_neighborenergy_coeff'] > upper_bound)
]

# Plot coefficients within the specified range
plt.figure(figsize=(10, 6))
plt.scatter(coefficients_within_range1['fips_code'], coefficients_within_range1['energy_neighborenergy_coeff'], 
            color='blue', label='Within {} Std Dev'.format(num_std_devs))
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Regression Coefficients within {} Standard Deviations from Mean'.format(num_std_devs))
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot coefficients outside the specified range
plt.figure(figsize=(10, 6))
plt.scatter(coefficients_outside_range1['fips_code'], coefficients_outside_range1['energy_neighborenergy_coeff'], 
            color='blue', label='Outside {} Std Dev'.format(num_std_devs))
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Regression Coefficients outside {} Standard Deviations from Mean'.format(num_std_devs))
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()



# In[160]:


# Plot the choropleth map of counties within the specified range
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
coefficients_within_range1.plot(ax=ax, edgecolor='black', cmap='coolwarm')
ax.set_title('Counties within {} Standard Deviations'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# In[161]:


# Plot the choropleth map of counties outside the specified range
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
coefficients_outside_range1.plot(ax=ax, edgecolor='black', cmap='coolwarm')
ax.set_title('Counties outside {} Standard Deviations'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# In[162]:


# Calculate mean of coefficients
coeff_mean1 = dfaisus1['energy_neighborenergy_coeff'].mean()

# Plot the distribution of coefficients
plt.figure(figsize=(10, 6))
plt.hist(dfaisus1['energy_neighborenergy_coeff'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(coeff_mean1, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title('Distribution of Regression Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.show()

# # Plot of the distribution for regression coefficients
# plot = plt.hist(dfaisus_energy['energy_neighborenergy_coeff'], bins=20, edgecolor='black')
# plt.axvline(np.mean(dfaisus_energy['energy_neighborenergy_coeff']), color='red', linestyle='dashed', linewidth=2)
# plt.xlabel('Regression Coefficient')
# plt.ylabel('Frequency')
# plt.title('Distribution of Regression Coefficients')


# min_value = np.min(dfaisus_energy['energy_neighborenergy_coeff'])
# max_value = np.max(dfaisus_energy['energy_neighborenergy_coeff'])
# plt.xlim(min_value, max_value)


# In[163]:


regression_data = dfaisus1

# Calculate mean and standard deviation of coefficients
coeff_mean1 = regression_data['energy_neighborenergy_coeff'].mean()
coeff_std1 = regression_data['energy_neighborenergy_coeff'].std()

# Define threshold values as 2 standard deviations from the mean
large_threshold1 = coeff_mean1 + 2 * coeff_std1
small_threshold1 = coeff_mean1 - 2 * coeff_std1

# Categorize coefficients
regression_data['Category'] = 'Other'
regression_data.loc[regression_data['energy_neighborenergy_coeff'] > large_threshold1, 'Category'] = 'Large Positive'
regression_data.loc[(regression_data['energy_neighborenergy_coeff'] <= large_threshold1) & (regression_data['energy_neighborenergy_coeff'] > 0), 'Category'] = 'Small Positive'
regression_data.loc[regression_data['energy_neighborenergy_coeff'] < small_threshold1, 'Category'] = 'Large Negative'
regression_data.loc[(regression_data['energy_neighborenergy_coeff'] >= small_threshold1) & (regression_data['energy_neighborenergy_coeff'] < 0), 'Category'] = 'Small Negative'

# Separate the data based on categories
large_positive_coeffs1 = regression_data[regression_data['Category'] == 'Large Positive']
small_positive_coeffs1 = regression_data[regression_data['Category'] == 'Small Positive']
large_negative_coeffs1 = regression_data[regression_data['Category'] == 'Large Negative']
small_negative_coeffs1 = regression_data[regression_data['Category'] == 'Small Negative']

# Create scatter plots
plt.figure(figsize=(10, 6))
plt.scatter(large_positive_coeffs1['fips_code'], large_positive_coeffs1['energy_neighborenergy_coeff'], 
            color='green', marker='o', label='Large Positive')
plt.scatter(small_positive_coeffs1['fips_code'], small_positive_coeffs1['energy_neighborenergy_coeff'], 
            color='lightgreen', marker='o', label='Small Positive')
plt.scatter(large_negative_coeffs1['fips_code'], large_negative_coeffs1['energy_neighborenergy_coeff'], 
            color='red', marker='x', label='Large Negative')
plt.scatter(small_negative_coeffs1['fips_code'], small_negative_coeffs1['energy_neighborenergy_coeff'], 
            color='salmon', marker='x', label='Small Negative')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot of Coefficients by County and Category')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[ ]:





# In[164]:


# Calculate the count of each category
category_counts = regression_data['Category'].value_counts()

# Create a pie chart
plt.figure(figsize=(12, 12))
colors = ['green', 'red', 'lightgreen', 'salmon']
plt.pie(category_counts, labels=category_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Coefficient Categories')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[165]:


# Create a histogram of coefficient categories
plt.figure(figsize=(10, 6))
category_order = ['Small Positive', 'Small Negative', 'Large Positive', 'Large Negative']
plt.hist(regression_data['Category'], bins=len(category_order), rwidth=0.8, align='left', color='skyblue', edgecolor='black')
plt.title('Histogram of Coefficient Categories')
plt.xlabel('Coefficient Category')
plt.ylabel('Frequency')
plt.xticks(range(len(category_order)), category_order, rotation=45)
plt.grid(axis='y')
plt.show()


# In[ ]:





# GDP vs Neighbors Energy

# In[166]:


# dfaisus_energy = map_gdpenergy.reset_index()


# In[167]:


# dfaisus_energy=dfaisus_energy.rename(columns={'gdp_neighborenergy_coeff': 'energy_neighborenergy_coeff'})


# In[168]:


# Calculate whether the coefficients are positive or negative
dfaisus1['sign_gdpenergy'] = np.where(dfaisus1['gdp_neighborenergy_coeff'] > 0, 'Positive', 'Negative')

# Count the occurrences of positive and negative coefficients
coeff_counts = dfaisus1['sign_gdpenergy'].value_counts()

# Create a pie chart
plt.figure(figsize=(12, 12))
colors = ['green', 'red']
plt.pie(coeff_counts, labels=coeff_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
#plt.title('Distribution of Positive and Negative Coefficients')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[169]:


# Separate positive and negative coefficients
positive_coeffs = dfaisus1[dfaisus1['gdp_neighborenergy_coeff'] > 0]
negative_coeffs = dfaisus1[dfaisus1['gdp_neighborenergy_coeff'] < 0]

# Create scatter plots
plt.figure(figsize=(10, 6))
plt.scatter(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborenergy_coeff'], 
            color='green', label='Positive Coefficients')
plt.scatter(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborenergy_coeff'], 
            color='red', label='Negative Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot of Positive and Negative Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot positive coefficients
plt.figure(figsize=(10, 6))
plt.scatter(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborenergy_coeff'], 
            color='green', label='Positive Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Positive Regression Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot negative coefficients
plt.figure(figsize=(10, 6))
plt.scatter(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborenergy_coeff'], 
            color='red', label='Negative Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Negative Regression Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# Plot the choropleth map of counties within positive coefficients
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
positive_coeffs.plot(ax=ax, edgecolor='black', cmap='coolwarm', legend='True')
ax.set_title('Counties with positive regression coefficients'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# Plot the choropleth map of counties within positive coefficients
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
negative_coeffs.plot(ax=ax, edgecolor='black', cmap='coolwarm')
ax.set_title('Counties with negative regression coefficients'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# In[170]:


# Create scatter plots with regression coefficients on the x-axis
plt.figure(figsize=(10, 6))

# Scatter plot for positive coefficients
plt.scatter(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborenergy_coeff'], 
            color='green', label='Positive Coefficients')

# Scatter plot for negative coefficients
plt.scatter(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborenergy_coeff'], 
            color='red', label='Negative Coefficients')

# Horizontal lines for the regression coefficients
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
for index, row in positive_coeffs.iterrows():
    plt.plot([row['fips_code'], row['fips_code']], [0, row['gdp_neighborenergy_coeff']], 
             color='green', linestyle='-', linewidth=1)
for index, row in negative_coeffs.iterrows():
    plt.plot([row['fips_code'], row['fips_code']], [0, row['gdp_neighborenergy_coeff']], 
             color='red', linestyle='-', linewidth=1)

plt.title('Scatter Plot of Positive and Negative Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[171]:


# Create scatter plots with regression coefficients on the x-axis and trend lines
plt.figure(figsize=(10, 6))

# Scatter plot for positive coefficients
plt.scatter(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborenergy_coeff'], 
            color='green', label='Positive Coefficients')

# Scatter plot for negative coefficients
plt.scatter(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborenergy_coeff'], 
            color='red', label='Negative Coefficients')

# Trend line for positive coefficients
positive_trend_x = np.linspace(min(positive_coeffs['fips_code']), max(positive_coeffs['fips_code']), 100)
positive_trend_y = np.polyval(np.polyfit(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborenergy_coeff'], 1), 
                              positive_trend_x)
plt.plot(positive_trend_x, positive_trend_y, color='green', linestyle='-', linewidth=2)

# Trend line for negative coefficients
negative_trend_x = np.linspace(min(negative_coeffs['fips_code']), max(negative_coeffs['fips_code']), 100)
negative_trend_y = np.polyval(np.polyfit(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborenergy_coeff'], 1), 
                              negative_trend_x)
plt.plot(negative_trend_x, negative_trend_y, color='red', linestyle='-', linewidth=2)

plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot of Positive and Negative Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[172]:


# Calculate mean and standard deviation of coefficients
coeff_mean = dfaisus1['gdp_neighborenergy_coeff'].mean()
coeff_std = dfaisus1['gdp_neighborenergy_coeff'].std()

# Calculate the range of coefficients within some standard deviations
num_std_devs = 2  # Change this value as needed
lower_bound = coeff_mean - num_std_devs * coeff_std
upper_bound = coeff_mean + num_std_devs * coeff_std

# Filter coefficients within the specified range
# Filter coefficients within and outside the specified range
coefficients_within_range = dfaisus1[
    (dfaisus1['gdp_neighborenergy_coeff'] >= lower_bound) & (dfaisus1['gdp_neighborenergy_coeff'] <= upper_bound)
]
coefficients_outside_range = dfaisus1[
    (dfaisus1['gdp_neighborenergy_coeff'] < lower_bound) | (dfaisus1['gdp_neighborenergy_coeff'] > upper_bound)
]

# Plot coefficients within the specified range
plt.figure(figsize=(10, 6))
plt.scatter(coefficients_within_range['fips_code'], coefficients_within_range['gdp_neighborenergy_coeff'], 
            color='blue', label='Within {} Std Dev'.format(num_std_devs))
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Regression Coefficients within {} Standard Deviations from Mean'.format(num_std_devs))
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot coefficients outside the specified range
plt.figure(figsize=(10, 6))
plt.scatter(coefficients_outside_range['fips_code'], coefficients_outside_range['gdp_neighborenergy_coeff'], 
            color='blue', label='Outside {} Std Dev'.format(num_std_devs))
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Regression Coefficients outside {} Standard Deviations from Mean'.format(num_std_devs))
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()



# In[173]:


# Create scatter plot with a single trend line for the entire dataframe
plt.figure(figsize=(10, 6))

# Scatter plot for all coefficients
plt.scatter(dfaisus1['fips_code'], dfaisus1['gdp_neighborenergy_coeff'], color='blue', label='Coefficients')

# Trend line for all coefficients
trend_x = np.linspace(min(dfaisus1['fips_code']), max(dfaisus1['fips_code']), 100)
trend_y = np.polyval(np.polyfit(dfaisus1['fips_code'], dfaisus1['gdp_neighborenergy_coeff'], 1), trend_x)
plt.plot(trend_x, trend_y, color='blue', linestyle='-', linewidth=2)

plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot and Trend Line of Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[174]:


# Calculate mean of coefficients
coeff_mean = dfaisus1['gdp_neighborenergy_coeff'].mean()

# Plot the distribution of coefficients
plt.figure(figsize=(10, 6))
plt.hist(dfaisus1['gdp_neighborenergy_coeff'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(coeff_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title('Distribution of Regression Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.show()

# # Plot of the distribution for regression coefficients
# plot = plt.hist(dfaisus_energy['energy_neighborenergy_coeff'], bins=20, edgecolor='black')
# plt.axvline(np.mean(dfaisus_energy['energy_neighborenergy_coeff']), color='red', linestyle='dashed', linewidth=2)
# plt.xlabel('Regression Coefficient')
# plt.ylabel('Frequency')
# plt.title('Distribution of Regression Coefficients')


# min_value = np.min(dfaisus_energy['energy_neighborenergy_coeff'])
# max_value = np.max(dfaisus_energy['energy_neighborenergy_coeff'])
# plt.xlim(min_value, max_value)


# In[175]:


# Calculate mean and standard deviation of coefficients
coeff_mean = regression_data['gdp_neighborenergy_coeff'].mean()
coeff_std = regression_data['gdp_neighborenergy_coeff'].std()

# Define threshold values as 2 standard deviations from the mean
large_threshold = coeff_mean + 2 * coeff_std
small_threshold = coeff_mean - 2 * coeff_std

# Categorize coefficients
regression_data['Category1'] = 'Other'
regression_data.loc[regression_data['gdp_neighborenergy_coeff'] > large_threshold, 'Category1'] = 'Large Positive'
regression_data.loc[(regression_data['gdp_neighborenergy_coeff'] <= large_threshold) & (regression_data['gdp_neighborenergy_coeff'] > 0), 'Category1'] = 'Small Positive'
regression_data.loc[regression_data['gdp_neighborenergy_coeff'] < small_threshold, 'Category1'] = 'Large Negative'
regression_data.loc[(regression_data['gdp_neighborenergy_coeff'] >= small_threshold) & (regression_data['gdp_neighborenergy_coeff'] < 0), 'Category1'] = 'Small Negative'

# Separate the data based on categories
large_positive_coeffs = regression_data[regression_data['Category1'] == 'Large Positive']
small_positive_coeffs = regression_data[regression_data['Category1'] == 'Small Positive']
large_negative_coeffs = regression_data[regression_data['Category1'] == 'Large Negative']
small_negative_coeffs = regression_data[regression_data['Category1'] == 'Small Negative']

# Create scatter plots
plt.figure(figsize=(10, 6))
plt.scatter(large_positive_coeffs['fips_code'], large_positive_coeffs['gdp_neighborenergy_coeff'], 
            color='green', marker='o', label='Large Positive')
plt.scatter(small_positive_coeffs['fips_code'], small_positive_coeffs['gdp_neighborenergy_coeff'], 
            color='lightgreen', marker='o', label='Small Positive')
plt.scatter(large_negative_coeffs['fips_code'], large_negative_coeffs['gdp_neighborenergy_coeff'], 
            color='red', marker='x', label='Large Negative')
plt.scatter(small_negative_coeffs['fips_code'], small_negative_coeffs['gdp_neighborenergy_coeff'], 
            color='salmon', marker='x', label='Small Negative')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot of Coefficients by County and Category')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[176]:


# Calculate the count of each category
category_counts = regression_data['Category1'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
colors = ['green', 'red', 'lightgreen', 'salmon']
plt.pie(category_counts, labels=category_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Coefficient Categories')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[177]:


# # Create a histogram of coefficient categories
# plt.figure(figsize=(10, 6))
# category_order = ['Large Positive', 'Small Positive', 'Small Negative', 'Large Negative']
# plt.hist(regression_data['Category1'], bins=len(category_order), rwidth=0.8, align='left', color='skyblue', edgecolor='black')
# plt.title('Histogram of Coefficient Categories')
# plt.xlabel('Coefficient Category')
# plt.ylabel('Frequency')
# plt.xticks(range(len(category_order)), category_order, rotation=45)
# plt.grid(axis='y')
# plt.show()


# GDP vs Neighbors GDP

# In[178]:


# dfaisus_energy = map_gdpgdp.reset_index()


# In[179]:


# dfaisus_energy=dfaisus_energy.rename(columns={'gdp_neighborgdp_coeff': 'energy_neighborenergy_coeff'})


# In[ ]:





# In[180]:


# Calculate whether the coefficients are positive or negative
dfaisus1['sign_gdpgdp'] = np.where(dfaisus1['gdp_neighborgdp_coeff'] > 0, 'Positive', 'Negative')

# Count the occurrences of positive and negative coefficients
coeff_counts = dfaisus1['sign_gdpgdp'].value_counts()

# Create a pie chart
plt.figure(figsize=(12, 12))
colors = ['green', 'red']
plt.pie(coeff_counts, labels=coeff_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
#plt.title('Distribution of Positive and Negative Coefficients')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[181]:


# Separate positive and negative coefficients
positive_coeffs = dfaisus1[dfaisus1['gdp_neighborgdp_coeff'] > 0]
negative_coeffs = dfaisus1[dfaisus1['gdp_neighborgdp_coeff'] < 0]

# Create scatter plots
plt.figure(figsize=(10, 6))
plt.scatter(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborgdp_coeff'], 
            color='green', label='Positive Coefficients')
plt.scatter(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborgdp_coeff'], 
            color='red', label='Negative Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot of Positive and Negative Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot positive coefficients
plt.figure(figsize=(10, 6))
plt.scatter(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborgdp_coeff'], 
            color='green', label='Positive Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Positive Regression Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot negative coefficients
plt.figure(figsize=(10, 6))
plt.scatter(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborgdp_coeff'], 
            color='red', label='Negative Coefficients')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Negative Regression Coefficients by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# Plot the choropleth map of counties within positive coefficients
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
positive_coeffs.plot(ax=ax, edgecolor='black', cmap='coolwarm')
ax.set_title('Counties with positive regression coefficients'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# Plot the choropleth map of counties within positive coefficients
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
negative_coeffs.plot(ax=ax, edgecolor='black', cmap='coolwarm')
ax.set_title('Counties with negative regression coefficients'.format(num_std_devs))
ax.set_axis_off()
plt.show()


# In[ ]:





# In[182]:


# Merge the two dataframes based on 'fips_code'
# merged_df = df1.merge(df2, on='fips_code', suffixes=('_df1', '_df2'))


# In[183]:


full_datapoint = energy_df['Diff']


# In[184]:


full_datapoint = full_datapoint.reset_index()


# In[185]:


full_datapoint = full_datapoint.rename(columns={'FIPS_y':'fips_code'})


# In[186]:


dfaisus2 = dfaisus1


# In[187]:


merge_full = full_datapoint.merge(dfaisus2, on='fips_code', suffixes=('full_datapoint', 'dfaisus2'))


# In[188]:


merge_full2 = merge_full


# In[189]:


# Separate positive and negative coefficients
positive_coeffs = dfaisus1[dfaisus1['gdp_neighborgdp_coeff'] > 0]
negative_coeffs = dfaisus1[dfaisus1['gdp_neighborgdp_coeff'] < 0]

# Calculate the overall trend line for coefficients
coeff_trend_x = np.linspace(min(dfaisus1['fips_code']), max(dfaisus1['fips_code']), 100)
coeff_trend_y = np.polyval(np.polyfit(dfaisus1['fips_code'], dfaisus1['gdp_neighborgdp_coeff'], 1), coeff_trend_x)
another_trend_y = np.polyval(np.polyfit(dfaisus1['fips_code'], dfaisus1['energy_neighborenergy_coeff'], 1), coeff_trend_x)
trend_full = np.polyval(np.polyfit(full_datapoint['fips_code'], full_datapoint['All industry total_y'], 1), coeff_trend_x)

# Create scatter plots
plt.figure(figsize=(40, 24))

# Scatter plot for positive coefficients
plt.scatter(positive_coeffs['fips_code'], positive_coeffs['gdp_neighborgdp_coeff'], 
            color='green', label='Positive Coefficients')

# Scatter plot for negative coefficients
plt.scatter(negative_coeffs['fips_code'], negative_coeffs['gdp_neighborgdp_coeff'], 
            color='red', label='Negative Coefficients')

# Plot the overall trend line for coefficients
plt.plot(coeff_trend_x, coeff_trend_y, color='blue', label='gdp vs gdp Trend', linestyle='-', linewidth=2)

# Plot another trend line for energy vs energy column
plt.plot(coeff_trend_x, another_trend_y, color='orange', label='energy vs energy Trend', linestyle='-', linewidth=2)

# Plot another trend line for full df
plt.plot(coeff_trend_x, trend_full, color='purple', label='full data points', linestyle='-', linewidth=2)

plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot and Coefficient Trend by County')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[190]:


merge_full2


# In[191]:


merge_full2['xbeta'] = merge_full2['NeighborGDP'] * merge_full2['gdp_neighborgdp_coeff']


# In[192]:


merge_full2


# In[193]:


len(merge_full['All industry total_y'])


# In[194]:


len(np.random.uniform(0, 10, 46438))


# In[195]:


import matplotlib.colors as mcolors

# Create scatter plot
# plt.figure(figsize=(10, 6))

# # Mapping fips_code to a continuous range from 0 to 50
# continuous_x = np.linspace(0, 100, len(merge_full['fips_code']))

# # Scatter plot for full data GDP
# plt.scatter(continuous_x, merge_full['All industry total_y'], color='blue', label='GDP')

# # Scatter plot for beta estimates
# plt.scatter(continuous_x, merge_full['gdp_neighborenergy_coeff'], color='red', label='Beta estimates')

# plt.title('Scatter Plot Comparison between GDP and Neighbor GDP')
# plt.xlabel('values')
# plt.ylabel('Estimates & GDP')
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.show()




# Sample data - Replace with your actual data
# Assume 'x_data' and 'y_data' are the continuous variables
x_data = merge_full['energy']#np.random.uniform(0, 50, 2756)  # Continuous variable for x-axis
y_data = merge_full['Neighborenergy']#np.random.uniform(0, 100, 2756)  # Continuous variable for y-axis
color_data = merge_full['energy_neighborenergy_coeff']  # Numeric values for color

# Create scatter plot
plt.figure(figsize=(20, 12))

# Scatter plot with color reflecting the numeric value of parameter estimate
scatter = plt.scatter(x_data, y_data, c=color_data, cmap='viridis', label='Scatter Plot')

# Adding a colorbar to indicate the color scale
plt.colorbar(scatter, label='Parameter Estimate')

plt.title('Scatter Plot with Color Reflecting Parameter Estimate')
plt.xlabel('energy')
plt.ylabel('Neighbor energy')
plt.legend()
plt.grid()

plt.show()


# In[196]:


import matplotlib.colors as mcolors

# Sample data - Replace with your actual data
x_data = merge_full['energy']
y_data = merge_full['Neighborenergy']
color_data = merge_full['energy_neighborenergy_coeff']

# Define the colormap and normalize the color data
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=-4, vmax=4)

# Create scatter plot
plt.figure(figsize=(20, 12))

# Scatter plot with color reflecting the numeric value of parameter estimate
scatter = plt.scatter(x_data, y_data, c=color_data, cmap=cmap, norm=norm, label='Scatter Plot')

# Creating a custom colorbar
cbar = plt.colorbar(scatter, label='Parameter Estimate')
cbar.set_ticks([-4, 0, 4])  # Set tick positions
cbar.set_ticklabels(['-4', '0', '4'])  # Set tick labels

plt.title('Scatter Plot with Color Reflecting Parameter Estimate')
plt.xlabel('energy')
plt.ylabel('Neighbor energy')
plt.legend()
plt.grid()

plt.show()


# In[197]:


import numpy as np
import matplotlib.pyplot as plt

# Sample data - Replace with your actual data
# Assume 'x_data' and 'y_data' are the continuous variables
x_data = np.random.uniform(0, 50, 100)  # Continuous variable for x-axis
y_data = np.random.uniform(0, 100, 100)  # Continuous variable for y-axis
color_data = np.random.uniform(0, 1, 100)  # Numeric values for color

# Create scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot with color reflecting the numeric value of parameter estimate
scatter = plt.scatter(x_data, y_data, c=color_data, cmap='viridis', label='Scatter Plot')

# Adding a colorbar to indicate the color scale
plt.colorbar(scatter, label='Parameter Estimate')

plt.title('Scatter Plot with Color Reflecting Parameter Estimate')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()

plt.show()


# In[198]:


# Calculate mean and standard deviation of coefficients
coeff_mean = dfaisus1['gdp_neighborgdp_coeff'].mean()
coeff_std = dfaisus1['gdp_neighborgdp_coeff'].std()

# Calculate the range of coefficients within some standard deviations
num_std_devs = 2  # Change this value as needed
lower_bound = coeff_mean - num_std_devs * coeff_std
upper_bound = coeff_mean + num_std_devs * coeff_std

# Filter coefficients within the specified range
# Filter coefficients within and outside the specified range
coefficients_within_range = dfaisus1[
    (dfaisus1['gdp_neighborgdp_coeff'] >= lower_bound) & (dfaisus1['gdp_neighborgdp_coeff'] <= upper_bound)
]
coefficients_outside_range = dfaisus1[
    (dfaisus1['gdp_neighborgdp_coeff'] < lower_bound) | (dfaisus1['gdp_neighborgdp_coeff'] > upper_bound)
]

# Plot coefficients within the specified range
plt.figure(figsize=(10, 6))
plt.scatter(coefficients_within_range['fips_code'], coefficients_within_range['gdp_neighborgdp_coeff'], 
            color='blue', label='Within {} Std Dev'.format(num_std_devs))
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Regression Coefficients within {} Standard Deviations from Mean'.format(num_std_devs))
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Plot coefficients outside the specified range
plt.figure(figsize=(10, 6))
plt.scatter(coefficients_outside_range['fips_code'], coefficients_outside_range['gdp_neighborgdp_coeff'], 
            color='blue', label='Outside {} Std Dev'.format(num_std_devs))
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Regression Coefficients outside {} Standard Deviations from Mean'.format(num_std_devs))
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()



# In[199]:


# Calculate mean of coefficients
coeff_mean = dfaisus1['gdp_neighborgdp_coeff'].mean()

# Plot the distribution of coefficients
plt.figure(figsize=(10, 6))
plt.hist(dfaisus1['gdp_neighborgdp_coeff'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(coeff_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title('Distribution of Regression Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.show()

# # Plot of the distribution for regression coefficients
# plot = plt.hist(dfaisus_energy['energy_neighborenergy_coeff'], bins=20, edgecolor='black')
# plt.axvline(np.mean(dfaisus_energy['energy_neighborenergy_coeff']), color='red', linestyle='dashed', linewidth=2)
# plt.xlabel('Regression Coefficient')
# plt.ylabel('Frequency')
# plt.title('Distribution of Regression Coefficients')


# min_value = np.min(dfaisus_energy['energy_neighborenergy_coeff'])
# max_value = np.max(dfaisus_energy['energy_neighborenergy_coeff'])
# plt.xlim(min_value, max_value)


# In[200]:


regression_data = dfaisus1

# Calculate mean and standard deviation of coefficients
coeff_mean = regression_data['gdp_neighborgdp_coeff'].mean()
coeff_std = regression_data['gdp_neighborgdp_coeff'].std()

# Define threshold values as 2 standard deviations from the mean
large_threshold = coeff_mean + 2 * coeff_std
small_threshold = coeff_mean - 2 * coeff_std

# Categorize coefficients
regression_data['Category2'] = 'Other'
regression_data.loc[regression_data['gdp_neighborgdp_coeff'] > large_threshold, 'Category2'] = 'Large Positive'
regression_data.loc[(regression_data['gdp_neighborgdp_coeff'] <= large_threshold) & (regression_data['gdp_neighborgdp_coeff'] > 0), 'Category2'] = 'Small Positive'
regression_data.loc[regression_data['gdp_neighborgdp_coeff'] < small_threshold, 'Category2'] = 'Large Negative'
regression_data.loc[(regression_data['gdp_neighborgdp_coeff'] >= small_threshold) & (regression_data['gdp_neighborgdp_coeff'] < 0), 'Category2'] = 'Small Negative'

# Separate the data based on categories
large_positive_coeffs = regression_data[regression_data['Category2'] == 'Large Positive']
small_positive_coeffs = regression_data[regression_data['Category2'] == 'Small Positive']
large_negative_coeffs = regression_data[regression_data['Category2'] == 'Large Negative']
small_negative_coeffs = regression_data[regression_data['Category2'] == 'Small Negative']

# Create scatter plots
plt.figure(figsize=(10, 6))
plt.scatter(large_positive_coeffs['fips_code'], large_positive_coeffs['gdp_neighborgdp_coeff'], 
            color='green', marker='o', label='Large Positive')
plt.scatter(small_positive_coeffs['fips_code'], small_positive_coeffs['gdp_neighborgdp_coeff'], 
            color='lightgreen', marker='o', label='Small Positive')
plt.scatter(large_negative_coeffs['fips_code'], large_negative_coeffs['gdp_neighborgdp_coeff'], 
            color='red', marker='x', label='Large Negative')
plt.scatter(small_negative_coeffs['fips_code'], small_negative_coeffs['gdp_neighborgdp_coeff'], 
            color='salmon', marker='x', label='Small Negative')
plt.axhline(0, color='black', linewidth=0.8, linestyle='dashed')
plt.title('Scatter Plot of Coefficients by County and Category')
plt.xlabel('County')
plt.ylabel('Coefficient')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[201]:


# Calculate the count of each category
category_counts = regression_data['Category2'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
colors = ['lightgreen', 'salmon', 'green', 'red']
plt.pie(category_counts, labels=category_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Coefficient Categories')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[202]:


# # Create a histogram of coefficient categories
# plt.figure(figsize=(10, 6))
# category_order = ['Large Positive', 'Small Positive', 'Small Negative', 'Large Negative']
# plt.hist(regression_data['Category2'], bins=len(category_order), rwidth=0.8, align='left', color='skyblue', edgecolor='black')
# plt.title('Histogram of Coefficient Categories')
# plt.xlabel('Coefficient Category')
# plt.ylabel('Frequency')
# plt.xticks(range(len(category_order)), category_order, rotation=45)
# plt.grid(axis='y')
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Try looping to create map from a single dataframe

# In[203]:


# reg_df_map = reg_coeff_map.reset_index()

# reg_df_map = reg_df_map.rename(columns={'FIPS_y':'fips_code'})

# reg_df_map = reg_df_map.set_index('fips_code')

# map_data.reset_index().set_index('fips_code', inplace=True)

# reg_df_map = pd.merge(map_data, reg_df_map, left_index=True, right_on='fips_code')


# In[204]:


# # List of column names
# column_names = ['gdp_neighborgdp_coeff', 'gdp_neighborenergy_coeff', 'gdp_neighborenergy_coeff']

# # Calculate the maximum absolute value for each column
# max_values = reg_df_map[column_names].abs().max()

# # Get the overall maximum value
# max_value = max_values.max()

# # Loop through the columns
# for column in column_names:
#     fig = px.choropleth(
#         reg_df_map,
#         geojson=reg_df_map['geometry'],
#         locations=reg_df_map.index,
#         color=column,
#         color_continuous_scale='RdBu',
#         color_continuous_midpoint=0,
#         range_color=[-max_value, max_value]
#     )
#     fig.update_geos(fitbounds='locations', visible=False)
#     fig.show()


# In[205]:


# column_names = ['gdp_neighborgdp_coeff', 'gdp_neighborenergy_coeff', 'gdp_neighborenergy_coeff']

# # Loop through the columns
# for column in column_names:
#     fig = px.choropleth(reg_df_map, geojson=reg_df_map['geometry'], locations=reg_df_map.index, color=column)
#     fig.update_geos(fitbounds='locations', visible=False)
#     fig.show()


# In[ ]:





# In[ ]:





# In[206]:


corr.reset_index(inplace=True)

corr_pop.reset_index(inplace=True)


# In[207]:


corr= corr.rename(columns={'FIPS_y': 'fips_code', 'NeighborGDP_y':'corr_GDP'})

corr_pop = corr_pop.rename(columns={'FIPS_y': 'fips_code', 'NeighborGDP_y':'corr_GDP_pop'})


# In[208]:


map_data.reset_index().set_index('fips_code', inplace=True)


# In[209]:


tdf = pd.merge(map_data, corr_pop, left_index=True, right_on='fips_code')


# In[210]:


fig, ax = plt.subplots(figsize = (26,16))
tdf.plot(column = "ownGDP_pop", ax = ax)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## AI update

# In[211]:


df1 = df.copy().reset_index().set_index("TimePeriod")
df1 = df1.join(divisia).reset_index()     
df1 = df1.set_index(["FIPS_y", "TimePeriod"])    
df1 = df1.sort_index()
df1


# In[212]:


df2_key = ['Agriculture, forestry, fishing and hunting_y',
 'Mining, quarrying, and oil and gas extraction_y',
 'Utilities_y',
 "All industry total_y", 'NeighborGDP_y', 'GDP_weigh', 'M4', 'M4 Interest Rate']
df2 = df1[df2_key]


# In[213]:


df3 = {}
df3["Log data"] = np.log(df2).replace([np.inf, -np.inf], np.nan)
df3["Diff"] = df3["Log data"].groupby("FIPS_y").diff().dropna()
df3["Diff2"] = df3["Diff"].groupby("FIPS_y").diff() 
df3


# In[ ]:





# In[214]:


def ips_test(data):
    index_name, sub_index_name = data.index.names
    index = list(data.reset_index()[index_name].unique())
    N = len(index)
    t_stats = {}
    df_index_dict = {}
    for ix in index:
        slice_df = data.loc[ix]
#         print(slice_df)
        t_stats[ix] = {}
        for key in slice_df.keys():
            try:
                t_stat = adfuller(slice_df[key], maxlag = 1, regression = 'c')[0]
                t_stats[ix][key] = t_stat
            except:
                print("Error:", key)
    t_stats = pd.DataFrame(t_stats).T

    return t_stats.mean()
   
ips_results = {}
for key, val in df3.items():
    ips_results[key] = ips_test(val.dropna())
pd.DataFrame(ips_results).dropna()


# In[ ]:





# In[215]:


plt.rcParams.update({'font.size': 30})
plt.rcParams['axes.xmargin'] = .001
plt.rcParams['axes.ymargin'] = .005
def full_corr_plot(data, color = "C0", pcorr = False):
    if pcorr == True:
        corr_df = data.pcorr()
    elif pcorr == False:
        corr_df = data.corr()
    keys = list(corr_df.keys())
    dim = len(keys)

    fig, ax = plt.subplots(figsize = (30, 30))
    a = pd.plotting.scatter_matrix(data, c = color, 
                                   s = 200, alpha = .1, ax=ax)  
    for i in range(len(keys)):
        x = keys[i]
        for j in range(len(keys)):
            y = keys[j]
            a[i][j].set_xticklabels([])
            a[i][j].set_yticklabels([])
            a[i][j].set_title("$\\rho :" + str(corr_df.round(2)[x][y])+ "$", y = .88, x = 0.01, ha = "left")        
    plt.suptitle("Correlation\n(Color: y)",y = .96, fontsize = 80)
df4 = df3['Diff2'].dropna()
df4.rename(columns = {key:key.replace(" ", "\n") for key in df4.keys()}, inplace = True)
df4_keys = list(df4.keys())
full_corr_plot(df4, color = df4[df4_keys[0]], pcorr = True)
# y_var = ['Agriculture, forestry, fishing and hunting']
# x_vars = ['Mining, quarrying, and oil and gas extraction', 'Utilities', 'Construction', 'Manufacturing']
# corr_var = y_var + x_vars
# corr_data = log_df[corr_var]
# corr_data.corr().round(3)


# In[216]:


import numpy as np
# . . .
def corr_matrix_heatmap(data, pp = False):  
    #Create a figure to visualize a corr matrix  
    fig, ax = plt.subplots(figsize=(20,20))  
    # use ax.imshow() to create a heatmap of correlation values  
    # seismic mapping shows negative values as blue and positive values as red  
    im = ax.imshow(data, norm = plt.cm.colors.Normalize(-1,1), cmap = "seismic")  
    # create a list of labels, stacking each word in a label by replacing " "  
    # with "\n"  
    labels = data.keys()  
    num_vars = len(labels)  
    tick_labels = [lab.replace(" ", "\n") for lab in labels]  
    # adjust font size according to the number of variables visualized  
    tick_font_size = 120 / num_vars  
    val_font_size = 200 / num_vars  
    plt.rcParams.update({'font.size': tick_font_size}) 
    # prepare space for label of each column  
    x_ticks = np.arange(num_vars)  
    # select labels and rotate them 90 degrees so that they are vertical  
    plt.xticks(x_ticks, tick_labels, fontsize = tick_font_size, rotation = 90)  
    # prepare space for label of each row  
    y_ticks = np.arange(len(labels))  
    # select labels  
    plt.yticks(y_ticks, tick_labels, fontsize = tick_font_size)  
    # show values in each tile of the heatmap  
    for i in range(len(labels)):  
        for j in range(len(labels)):  
            text = ax.text(i, j, str(round(data.values[i][j],2)),  
                           fontsize= val_font_size, ha="center",   
                           va="center", color = "w")  
    #Create title with Times New Roman Font  
    title_font = {"fontname":"Times New Roman"}  
    plt.title("Correlation", fontsize = 50, **title_font)  
    #Call scale to show value of colors 
    cbar = fig.colorbar(im)
    plt.show()
    if pp != False:
        pp.savefig(fig, bbox_inches="tight")
    plt.close()

#. . . 
# . . .
corr_matrix_heatmap(df4.corr())


# In[217]:


df4.rename(columns = {key:key[:4].replace("\n", "") for key in df4.keys()}, inplace = True)
list(df4.keys())


# In[218]:


df4


# In[219]:


from matplotlib.patches import ArrowStyle
import copy
from matplotlib.backends.backend_pdf import PdfPages

undirected_graph = {key:[] for key in df4.keys()}
for x in undirected_graph:
    remaining_vars = [y for y in df4.keys() if y != x]
    for y in remaining_vars:
        undirected_graph[x].append(y)

p_value = .01
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import PC
c = PC(df4)
max_cond_vars = len(df4.keys()) - 2

model = c.estimate(return_type = 'pdag', variant= 'parallel', significance_level = p_value,
                  max_cond_vars = max_cond_vars, ci_test = 'pearsonr')
edges = model.edges

pp = PdfPages("DAGOutputs1.pdf")

def graph_DAG(edges, df, title = ""):
    graph = nx.Graph()
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
        pcorr = df[[edge[0], edge[1]]+keep_controls].pcorr()
        edge_labels[edge] = str(round(pcorr[edge[0]].loc[edge[1]],2))
    graph.add_edges_from(edges)
    color_map = ['C0' for g in graph]
    
    fig, ax = plt.subplots(figsize = (20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize = 30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                    arrows=True, font_color ='k', font_size=26, alpha=1, width = 1,
                    edge_color = 'C1',
                     arrowstyle=ArrowStyle('Fancy, head_length=3, head_width=1.5, tail_width=.1'), ax = ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches = "tight")

graph_DAG(edges, df4, title = 'Directed Acyclic Graph')


pp.close()                                                            
edges


# In[220]:


# result = pd.concat([df1, df2], axis=0)


# In[221]:


# full_df.loc[year, year_data.index]


# In[222]:


# test_data


# In[223]:


# test_data = map_data.join(year_data).dropna(subset = ["All industry total"])
# find_neighbors(test_data)
# # year_data


# In[224]:


# for index, GeoFips in county_gdf.iterrows():
#     neighbors = county_gdf[~county_gdf.geometry.disjoint(GeoFips.geometry)].county_name.tolist()
#     neighbor = [name for name in neighbors if GeoFips.county_name != name]
#     county_gdf.at[index, "NEIGHBORS"] = ", ".join(neighbors)


# In[225]:


# county_gdf.to_file("C:/Users/abiodun.idowu/OneDrive - North Dakota University System/Desktop/PhD/BEA project/notebook_to_start/newfile.shp")


# In[226]:


#drop columns 9 to 24 
# gdf = gdf.drop(gdf.columns[9:24], axis=1)


# In[227]:


# county_gdf


# In[228]:


# merge_df1 = pd.merge(plot_df.reset_index(), county_gdf, left_on='GeoFips', right_on='GeoFips')


# In[229]:


# merge_df1 = merge_df1.set_index('GeoFips')


# In[230]:


# merge_df1


# In[231]:


# import geopandas as gpd

# fig, ax = plt.subplots(figsize=(26,16))
# county_gdf.plot(column='NEIGHBORS', ax=ax)


# In[232]:


# print(county_gdf.dtypes)


# In[233]:


# test_df = pd.merge(county_gdf.reset_index(), plot_df, left_on='GeoFips', right_on='GeoFips')


# In[234]:


# test_df = test_df.set_index('GeoFips')


# In[235]:


# test_df


# In[ ]:





# In[236]:


# #create a column with the sum of neighboring gdps

# for index, GeoFips in gdf.iterrows():
#     neighbors = gdf[~gdf.geometry.disjoint(GeoFips.geometry)].county_name.tolist()
#     neighbor = [name for name in neighbors if GeoFips.county_name != name]
    
#     neighboring_GDP = gdf[gdf.county_name.isin(neighbors)]["All"].sum()
#     gdf.at[index, "gdp_sum"] = neighboring_GDP


# In[237]:


# map_data.index


# In[238]:


# full_df = full_df.reset_index()
# full_df["FIPS"] = full_df["GeoFips"]
# full_df = full_df.set_index(["TimePeriod","GeoFips"])
# full_df.dropna(subset = ["All industry total"], inplace = True)


# In[239]:


# full_df


# In[ ]:





# In[240]:


# # neighbors = map_data.loc[21007,"NEIGHBORS"]

# full_df["NeighborGDP"] = np.NaN

# for year in range(2004,2020):
#     year_data = full_df.loc[year]
#     year_data = map_data.join(year_data).dropna(subset = ["All industry total"])
#     find_neighbors(year_data)    
# #     year_data = year_data.join(map_data["NEIGHBORS"])
# #     year_data["NeighborGDP"]
# #     print( year_data["All industry total"].loc[year_data.loc[1001]["NEIGHBORS"]].sum())
# #     full_df.loc[year, "All industry total"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
# #                           axis = 1)
#     full_df.loc[year, "All industry total"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
#                           axis = 1)
# #     full_df.loc[year, "NeighborGDP"] = year_data.apply(lambda row: year_data["All industry total"].loc[row["NEIGHBORS"]].sum() if row["NEIGHBORS"] != np.nan else np.nan, 
# #                           axis = 1)

    
#     #     gdf["NeighborGDP"][gdf["TimePeriod"] == year]] = year_data[year_data.county_name.isin(neighbors)]["All"].sum()

# # for row in year_data.iterrows():
# #     print(row[0])


# In[241]:


# full_df["All industry total"]


# In[242]:


# full_df["NeighborGDP"]


# In[243]:


# full_df.dropna(subset = ["All industry total"])


# In[244]:


#  map_data.loc[year_data.loc[1001]["FIPS"]]["NEIGHBORS"]


# In[245]:


# map_data.loc[1001]["NEIGHBORS"]


# In[246]:


# gdf["NeighborGDP"] = np.NaN
# for year in range(2004,2020):
#     year_data = gdf[gdf["TimePeriod"] == 2004]
#     gdf["NeighborGDP"][gdf["TimePeriod"] == 2004] = year_data[year_data.county_name.isin(neighbors)]["All"].sum()


# In[247]:


# gdf


# In[248]:


# !pip install pygeos


# In[ ]:





# In[249]:


# summing the gdp for neighbor per year


# In[ ]:





# In[250]:


df4


# In[251]:


dfg = df4


# In[252]:


dfg = dfg.reset_index()


# In[253]:


dfg


# In[254]:


# dfg.to_csv('stat_713b.csv')
# df_reg2.to_csv('stat_712.csv', index=False)


# In[255]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# In[256]:


silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dfg)
    silhouette_scores.append(silhouette_score(dfg, kmeans.labels_))

# Plot the silhouette scores to find the optimal k
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()


# In[257]:


dfg1 = dfg.drop(['M4', 'M4I'], axis=1)


# In[ ]:





# In[258]:


silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dfg1)
    silhouette_scores.append(silhouette_score(dfg1, kmeans.labels_))

# Plot the silhouette scores to find the optimal k
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()


# In[259]:


k_optimal = 4

kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(dfg)

# Add cluster labels to your DataFrame
df4['Cluster'] = kmeans.labels_


# In[260]:


df4


# In[ ]:





# In[261]:


# Group the DataFrame by 'Cluster' and calculate summary statistics for each cluster
data = df4

cluster_summary = data.groupby('Cluster').agg({
    'Agri': ['mean', 'std', 'count', 'min', 'max'],
    'Mini': ['mean', 'std', 'count', 'min', 'max'],
    'Util': ['mean', 'std', 'count', 'min', 'max'],
    'All': ['mean', 'std', 'count', 'min', 'max'],
    'Neig': ['mean', 'std', 'count', 'min', 'max'],
    'GDP_': ['mean', 'std', 'count', 'min', 'max'],
    'M4': ['mean', 'std', 'count', 'min', 'max'],
    'M4I': ['mean', 'std', 'count', 'min', 'max']
})

# Reset the column names for clarity
#cluster_summary.columns = ['Mean1', 'Std1', 'Count1', 'Min1', 'Max1', 'Mean2', 'Std2', 'Count2', 'Min2', 'Max2', ...]

# Display the summary statistics for each cluster
cluster_summary


# In[262]:


full_summary = data.agg({
    'Agri': ['mean', 'std', 'count', 'min', 'max'],
    'Mini': ['mean', 'std', 'count', 'min', 'max'],
    'Util': ['mean', 'std', 'count', 'min', 'max'],
    'All': ['mean', 'std', 'count', 'min', 'max'],
    'Neig': ['mean', 'std', 'count', 'min', 'max'],
    'GDP_': ['mean', 'std', 'count', 'min', 'max'],
    'M4': ['mean', 'std', 'count', 'min', 'max'],
    'M4I': ['mean', 'std', 'count', 'min', 'max']    
})

full_summary


# In[ ]:





# In[263]:


data_dfg = df4


# In[264]:


data_dfg = data_dfg.reset_index()


# In[265]:


data = data_dfg

# Select the features (attributes) for clustering
features = data[['Agri', 'Mini', 'Util', 'All', 'Neig', 'M4', 'M4I']].values

# Standardize the features (important for K-means)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Add cluster labels to the DataFrame
data['Cluster'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(10, 6))

# Scatterplot of the first two features colored by cluster
plt.scatter(data['Agri'], data['Mini'], c=cluster_labels, cmap='rainbow', s=50)

# Add cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Agri')
plt.ylabel('Mini')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[266]:


# Selected attributes for clustering
#features = data[['Agri', 'Mini', 'Util', 'All', 'Neig', 'GDP_, 'M4', 'M4I']]
#features = data[['Agri', 'Mini', 'Util', 'Neig', 'M4', 'M4I']]
#features = data[['Agri', 'Mini', 'Util', 'All','Neig']]
#features = data[['Agri', 'Mini', 'Util','All', 'Neig', 'M4', 'M4I']]
features = data[['Agri', 'Mini', 'Util','All', 'M4', 'M4I']]
                 
# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Visualize the clusters using pairplots, excluding 'timeperiod' and 'fips_codes'
sns.set(style="ticks")
pairplot_data = data.copy()
pairplot_data['Cluster'] = cluster_labels

# Exclude 'timeperiod' and 'fips_codes' columns from the pairplot
columns_to_exclude = ['TimePeriod', 'FIPS_y']
columns_to_include = [col for col in pairplot_data.columns if col not in columns_to_exclude]

sns.pairplot(pairplot_data[columns_to_include], hue='Cluster', palette='rainbow', plot_kws={'s': 50})
plt.suptitle('K-Means Clustering for Selected Variables')
plt.show()


# In[ ]:





# In[267]:


# Create a DataFrame with cluster labels and selected features
cluster_data = pd.DataFrame({'Cluster': cluster_labels})
cluster_data = pd.concat([cluster_data, features], axis=1)

# Calculate the correlation matrix
correlation_matrix = cluster_data.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(10, 8))
max_val = np.abs(correlation_matrix).max().max()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-max_val, vmax=max_val)
plt.title('Correlation Map of Clustering vs. Variables')
plt.show()


# In[ ]:





# In[268]:


from sklearn.linear_model import LinearRegression
# Create a DataFrame with cluster labels and selected features
cluster_data = pd.DataFrame({'Cluster': cluster_labels})
cluster_data = pd.concat([cluster_data, features], axis=1)

# Create a dictionary to store the regression results
regression_results = {}

# Iterating through each variable and performing linear regression, 
#the dependent variable is the cluster categories 

for column in features.columns:
    X = cluster_data[[column]]
    y = cluster_data['Cluster']
    
    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Store the regression coefficients in the dictionary
    regression_results[column] = {
        'Coefficient': model.coef_[0],
        'Intercept': model.intercept_,
        'R-squared': model.score(X, y)
    }

# Print the regression results
for variable, results in regression_results.items():
    print(f"Variable: {variable}")
    print(f"Coefficient: {results['Coefficient']:.4f}")
    print(f"Intercept: {results['Intercept']:.4f}")
    print(f"R-squared: {results['R-squared']:.4f}")
    print()


# With M4 variables

# In[269]:


# Create a DataFrame with cluster labels and selected features
features = data[['Agri', 'Mini', 'Util','All', 'M4', 'M4I']]
cluster_data = pd.DataFrame({'Cluster': cluster_labels})
cluster_data = pd.concat([cluster_data, features], axis=1)

# Create a dictionary to store the regression results
regression_results = {}

# Iterate through each cluster category
for cluster_category in cluster_data['Cluster'].unique():
    # Create a dictionary to store the regression results for the current cluster category
    cluster_results = {}
    
    # Iterate through each variable
    for variable in features.columns:
        # Perform a regression for the current cluster category against the current variable
        X = cluster_data[[variable]]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = (cluster_data['Cluster'] == cluster_category).astype(int)  # Binary outcome
        
        # Create and fit a logistic regression model
        model = sm.Logit(y, X).fit()
        
        # Store the regression coefficients in the dictionary
        cluster_results[variable] = {
            'Coefficient': model.params[variable],
            'Intercept': model.params['const'],
            'P-value': model.pvalues[variable]
        }
    
    # Store the cluster-specific regression results in the regression_results dictionary
    regression_results[cluster_category] = cluster_results

# Create a DataFrame to hold the regression coefficients for each cluster category and variable
heatmap_data = pd.DataFrame()

# Iterate through each cluster category and variable to construct the DataFrame
for cluster_category, cluster_results in regression_results.items():
    for variable, results in cluster_results.items():
        row_data = {
            'Cluster_Category': cluster_category,
            'Variable': variable,
            'Coefficient': results['Coefficient'],
            'P-value': results['P-value']
        }
        heatmap_data = heatmap_data.append(row_data, ignore_index=True)

# Pivot the DataFrame to prepare for the heatmap
heatmap_data_pivot = heatmap_data.pivot('Variable', 'Cluster_Category', 'Coefficient')

# Create a heatmap of the regression coefficients
plt.figure(figsize=(12, 8))
max_val = np.abs(heatmap_data_pivot).max().max()
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt=".4f", linewidths=0.5, vmin=max_val*-1, vmax=max_val)
plt.title('Logistic Regression Coefficients Heatmap between Cluster Categories and Variables')
plt.show()


# Without M4 variables

# In[270]:


# Create a DataFrame with cluster labels and selected features
features = data[['Agri', 'Mini', 'Util','All']]
cluster_data = pd.DataFrame({'Cluster': cluster_labels})
cluster_data = pd.concat([cluster_data, features], axis=1)

# Create a dictionary to store the regression results
regression_results = {}

# Iterate through each cluster category
for cluster_category in cluster_data['Cluster'].unique():
    # Create a dictionary to store the regression results for the current cluster category
    cluster_results = {}
    
    # Iterate through each variable
    for variable in features.columns:
        # Perform a regression for the current cluster category against the current variable
        X = cluster_data[[variable]]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = (cluster_data['Cluster'] == cluster_category).astype(int)  # Binary outcome
        
        # Create and fit a logistic regression model
        model = sm.Logit(y, X).fit()
        
        # Store the regression coefficients in the dictionary
        cluster_results[variable] = {
            'Coefficient': model.params[variable],
            'Intercept': model.params['const'],
            'P-value': model.pvalues[variable]
        }
    
    # Store the cluster-specific regression results in the regression_results dictionary
    regression_results[cluster_category] = cluster_results

# Create a DataFrame to hold the regression coefficients for each cluster category and variable
heatmap_data = pd.DataFrame()

# Iterate through each cluster category and variable to construct the DataFrame
for cluster_category, cluster_results in regression_results.items():
    for variable, results in cluster_results.items():
        row_data = {
            'Cluster_Category': cluster_category,
            'Variable': variable,
            'Coefficient': results['Coefficient'],
            'P-value': results['P-value']
        }
        heatmap_data = heatmap_data.append(row_data, ignore_index=True)

# Pivot the DataFrame to prepare for the heatmap
heatmap_data_pivot = heatmap_data.pivot('Variable', 'Cluster_Category', 'Coefficient')

# Create a heatmap of the regression coefficients
plt.figure(figsize=(12, 8))
max_val = np.abs(heatmap_data_pivot).max().max()
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt=".4f", linewidths=0.5, vmin=max_val*-1, vmax=max_val)
plt.title('Logistic Regression Coefficients Heatmap between Cluster Categories and Variables')
plt.show()


# In[271]:


unique_data = data.drop_duplicates(subset='FIPS_y')


# In[272]:


unique_data.reset_index(drop=True, inplace=True)


# In[273]:


unique_data= unique_data.rename(columns={'FIPS_y': 'fips_code'})


# In[274]:


cl_map = pd.merge(map_data, unique_data, left_index=True, right_on='fips_code')


# In[ ]:





# In[275]:


# cl_map.keys()


# In[276]:


fig = px.choropleth(cl_map, 
                    geojson=cl_map.geometry,  # Use the geometries as the geojson
                    locations=cl_map.index,  # Use the index of the GeoDataFrame
                    color="Cluster",  # Cluster labels as color
                    scope="usa"  # Set the scope to the USA map
                   )

fig.update_geos(
    visible=False,  # Hide the world map
    showcoastlines=True,  # Show coastlines on the map
)

fig.update_layout(
    title="K-Means Clustering of Counties",
    geo=dict(
        center={"lat": 37.0902, "lon": -95.7129},  # Set the center of the map (approximate center of the USA)
        projection_scale=6,  # Adjust the scale for the map
    )
)

fig.show()

#fig.write_html('cluster_map.html')


# In[277]:


hover_data = ['fips_code', "state", "NAME"]
fig = px.choropleth(cl_map, 
                    geojson=cl_map.geometry,  # Use the geometries as the geojson
                    locations=cl_map.index,  # Use the index of the GeoDataFrame
                    color="Cluster",  # Cluster labels as color
                    scope="usa", hover_data=hover_data 
                   )

fig.update_geos(
    visible=False,  # Hide the world map
    showcoastlines=True,  # Show coastlines on the map
)

fig.update_layout(
    title="K-Means Clustering of Counties",
    geo=dict(
        center={"lat": 37.0902, "lon": -95.7129},  # Set the center of the map (approximate center of the USA)
        projection_scale=6
    )
)

fig.show()

fig.write_html('cluster_map_detail.html')


# In[278]:


dfg_dag = data_dfg


# In[279]:


grouped = dfg_dag.groupby('Cluster')

# Initialize an empty dictionary to store the DataFrames for each cluster
cluster_dataframes = {}

# Iterate over the groups and create DataFrames
for cluster, group in grouped:
    cluster_dataframes[cluster] = group



# In[280]:


#Cluster dataframes
c1_df = cluster_dataframes[0]
c2_df = cluster_dataframes[1]
c3_df = cluster_dataframes[2]
c4_df = cluster_dataframes[3]


# In[281]:


c1_df


# In[282]:


c4_df


# In[283]:


dfg_dag = dfg_dag.set_index(['FIPS_y', 'TimePeriod'])


# In[ ]:





# In[284]:


list(dfg_dag.keys())


# In[ ]:





# In[285]:


c1_df.drop(['Cluster', 'Neig', 'GDP_'], axis=1, inplace=True)
c2_df.drop(['Cluster', 'Neig', 'GDP_'], axis=1, inplace=True)
c3_df.drop(['Cluster', 'Neig', 'GDP_'], axis=1, inplace=True)
c4_df.drop(['Cluster', 'Neig', 'GDP_'], axis=1, inplace=True)


# In[286]:


c1_df = c1_df.set_index(['FIPS_y', 'TimePeriod'])
c2_df = c2_df.set_index(['FIPS_y', 'TimePeriod'])
c3_df = c3_df.set_index(['FIPS_y', 'TimePeriod'])
c4_df = c4_df.set_index(['FIPS_y', 'TimePeriod'])


# In[ ]:





# In[ ]:





# In[287]:


# List of DataFrames
cluster_dataframes = [c1_df, c2_df, c3_df, c4_df]

p_value = 0.05
max_cond_vars = len(c1_df.columns) - 2  # Assuming all DataFrames have the same columns

pp = PdfPages("DAGOutputs_cluster2.pdf")


def graph_DAG(edges, df, title="Directed Acyclic Graph"):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.columns if key not in edge]
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
    color_map = ['C0' for g in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                     arrows=True, font_color='k', font_size=26, alpha=1, width=1,
                     edge_color='C1',
                     arrowstyle=ArrowStyle('Fancy, head_length=3, head_width=1.5, tail_width=.1'), ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches="tight")

for idx, c_df in enumerate(cluster_dataframes):
    c = PC(c_df)
    n = c_df.shape[0]
    model = c.estimate(return_type='pdag', variant='parallel', significance_level= p_value, max_cond_vars=max_cond_vars, ci_test='pearsonr')
    edges = model.edges

    # Call the graph_DAG function for each cluster DataFrame
    #graph_DAG(edges, c_df, title=f'Cluster {idx + 1} Directed Acyclic Graph')
    graph_DAG(edges, c_df, title=f'Cluster {idx + 1}\np_value = {p_value}\nnum_observations = {n}' )

pp.close()


# In[288]:


c_df.shape[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[289]:


max_val.max()


# In[290]:


# Create a DataFrame with cluster labels and selected features
cluster_data = pd.DataFrame({'Cluster': cluster_labels})
cluster_data = pd.concat([cluster_data, features], axis=1)

# Iterate through each unique cluster category
for cluster_category in cluster_data['Cluster'].unique():
    # Filter the data for the current cluster category
    cluster_category_data = cluster_data[cluster_data['Cluster'] == cluster_category]
    
    # Calculate the correlation matrix between variables and the current cluster category
    correlation_matrix = cluster_category_data.corr()
    
    # Create a heatmap to visualize the correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix.iloc[1:, :-1], annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Heatmap for Cluster Category {cluster_category}')
    plt.show()


# In[ ]:





# In[ ]:





# In[291]:


data = cl_map
# Get the unique cluster labels
unique_clusters = data['Cluster'].unique()

# Iterate through unique clusters and view unique entities within each cluster
for cluster_label in unique_clusters:
    cluster_data = data[data['Cluster'] == cluster_label]
    unique_entities = cluster_data['fips_code'].unique()
    print(f"Cluster {cluster_label} contains {len(unique_entities)} unique entities:")
    print(unique_entities)


# In[292]:


cluster_data = dfg.set_index(['FIPS_y', 'TimePeriod'])


# In[293]:


cluster_data.to_csv('cluster_data.csv')


# In[294]:


# !pip install tslearn


# Dynamic Time Warping

# In[295]:


from tslearn.clustering import TimeSeriesKMeans

dfg = cluster_data
# Define the number of clusters (K)
n_clusters = 4

# Perform time series clustering using DTW
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=True)
model.fit(dfg)
cluster_assignments = model.labels_
dfg['Cluster'] = cluster_assignments

# Select features for logistic regression
features = dfg.drop(['Cluster'], axis=1)

# Create a dictionary to store the regression results
regression_results = {}

# Iterate through each cluster category
for cluster_category in dfg['Cluster'].unique():
    # Create a dictionary to store the regression results for the current cluster category
    cluster_results = {}
    
    # Iterate through each variable
    for variable in features.columns:
        # Perform a regression for the current cluster category against the current variable
        X = features[[variable]]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = (dfg['Cluster'] == cluster_category).astype(int)  # Binary outcome
        
        # Create and fit a logistic regression model
        model = sm.Logit(y, X).fit()
        
        # Store the regression coefficients in the dictionary
        cluster_results[variable] = {
            'Coefficient': model.params[variable],
            'Intercept': model.params['const'],
            'P-value': model.pvalues[variable]
        }
    
    # Store the cluster-specific regression results in the regression_results dictionary
    regression_results[cluster_category] = cluster_results

# Create a DataFrame to hold the regression coefficients for each cluster category and variable
heatmap_data = pd.DataFrame()

# Iterate through each cluster category and variable to construct the DataFrame
for cluster_category, cluster_results in regression_results.items():
    for variable, results in cluster_results.items():
        row_data = {
            'Cluster_Category': cluster_category,
            'Variable': variable,
            'Coefficient': results['Coefficient'],
            'P-value': results['P-value']
        }
        heatmap_data = heatmap_data.append(row_data, ignore_index=True)

# Pivot the DataFrame to prepare for the heatmap
heatmap_data_pivot = heatmap_data.pivot('Variable', 'Cluster_Category', 'Coefficient')

# Create a heatmap of the regression coefficients
plt.figure(figsize=(12, 8))
max_val = np.abs(heatmap_data_pivot).max().max()
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt=".4f", linewidths=0.5, vmin=max_val*-1, vmax=max_val)
plt.title('Logistic Regression Coefficients Heatmap between Cluster Categories and Variables')
plt.show()


# In[ ]:





# In[296]:


grouped = cluster_data.groupby('Cluster')

# Initialize an empty dictionary to store the DataFrames for each cluster
cluster_dataframes = {}

# Iterate over the groups and create DataFrames
for cluster, group in grouped:
    cluster_dataframes[cluster] = group



# In[297]:


#Cluster dataframes
cl1_df = cluster_dataframes[0]
cl2_df = cluster_dataframes[1]
cl3_df = cluster_dataframes[2]
cl4_df = cluster_dataframes[3]


# ## Using kml3d clusters from R

# In[298]:


kml3d_cluster = pd.read_csv('kml3d_cluster_data.csv')


# In[299]:


kml3d_cluster


# In[300]:


kml3d_cluster = kml3d_cluster.reset_index()


# In[301]:


dfg = df4


# In[302]:


dfg = dfg.reset_index()


# In[303]:


dfg


# In[304]:


merged_df = pd.merge(dfg, kml3d_cluster[['FIPS_y', 'clusters']], on='FIPS_y', how='left')


# In[305]:


kml3d_cl_data = merged_df.dropna()


# In[306]:


kml3d_cl_data = kml3d_cl_data.set_index(['FIPS_y', 'TimePeriod'])


# In[307]:


kml3d_cl_data = kml3d_cl_data.drop('Cluster', axis=1)


# In[308]:


kml3d_cl_data


# In[309]:


cluster_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

kml3d_cl_data['clusters'] = kml3d_cl_data['clusters'].map(cluster_mapping)


# In[310]:


kml3d_cl_data


# In[311]:


features = kml3d_cl_data[['Agri', 'Mini', 'Util', 'All', 'M4', 'M4I']]


# In[312]:


# Create a dictionary to store the regression results
regression_results = {}

# Iterate through each cluster category
for cluster_category in kml3d_cl_data['clusters'].unique():
    # Create a dictionary to store the regression results for the current cluster category
    cluster_results = {}
    
    # Iterate through each variable
    for variable in features.columns:
        # Perform a regression for the current cluster category against the current variable
        X = features[[variable]]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = (kml3d_cl_data['clusters'] == cluster_category).astype(int)  # Binary outcome
        
        # Create and fit a logistic regression model
        model = sm.Logit(y, X).fit()
        
        # Store the regression coefficients in the dictionary
        cluster_results[variable] = {
            'Coefficient': model.params[variable],
            'Intercept': model.params['const'],
            'P-value': model.pvalues[variable]
        }
    
    # Store the cluster-specific regression results in the regression_results dictionary
    regression_results[cluster_category] = cluster_results

# Create a DataFrame to hold the regression coefficients for each cluster category and variable
heatmap_data = pd.DataFrame()

# Iterate through each cluster category and variable to construct the DataFrame
for cluster_category, cluster_results in regression_results.items():
    for variable, results in cluster_results.items():
        row_data = {
            'Cluster_Category': cluster_category,
            'Variable': variable,
            'Coefficient': results['Coefficient'],
            'P-value': results['P-value']
        }
        heatmap_data = heatmap_data.append(row_data, ignore_index=True)

# Pivot the DataFrame to prepare for the heatmap
heatmap_data_pivot = heatmap_data.pivot('Variable', 'Cluster_Category', 'Coefficient')

# Create a heatmap of the regression coefficients
plt.figure(figsize=(12, 8))
max_val = np.abs(heatmap_data_pivot).max().max()
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt=".4f", linewidths=0.5, vmin=max_val*-1, vmax=max_val)
plt.title('Logistic Regression Coefficients Heatmap between Cluster Categories and Variables')
plt.show()


# In[313]:


unique_kml3d = kml3d_cl_data.reset_index()


# In[314]:


unique_kml3d = unique_kml3d.drop_duplicates(subset='FIPS_y')


# In[315]:


unique_kml3d.reset_index(drop=True, inplace=True)


# In[316]:


unique_kml3d= unique_kml3d.rename(columns={'FIPS_y': 'fips_code'})


# In[317]:


unique_kml3d


# In[318]:


kml3d_map = pd.merge(map_data, unique_kml3d, left_index=True, right_on='fips_code')


# In[319]:


kml3d_map


# In[320]:


kml3d_map.index


# In[321]:


hover_data = ["state", "NAME"]
fig = px.choropleth(kml3d_map, 
                    geojson=kml3d_map['geometry'],  # Use the geometries as the geojson
                    locations=kml3d_map.index,  # Use the index of the GeoDataFrame
                    color="clusters",  # Cluster labels as color
                    scope="usa", hover_data=hover_data 
                   )

fig.update_geos(
    visible=False,  # Hide the world map
    showcoastlines=True,  # Show coastlines on the map
)

fig.update_layout(
    title="Clustering of Counties",
    geo=dict(
        center={"lat": 37.0902, "lon": -95.7129},  # Set the center of the map (approximate center of the USA)
        projection_scale=6
    )
)

fig.show()

fig.write_html('kml3d_cluster_map_detail.html')


# In[322]:


kml3d_map.columns


# In[323]:


kml3d_cl_data


# In[324]:


grouped = kml3d_cl_data.groupby('clusters')

# Initialize an empty dictionary to store the DataFrames for each cluster
cluster_dataframes = {}

# Iterate over the groups and create DataFrames
for cluster, group in grouped:
    cluster_dataframes[cluster] = group


# In[325]:


#Cluster dataframes
kml3d1_df = cluster_dataframes[1]
kml3d2_df = cluster_dataframes[2]
kml3d3_df = cluster_dataframes[3]
kml3d4_df = cluster_dataframes[4]


# In[326]:


kml3d1_df.drop(['clusters'], axis=1, inplace=True)
kml3d2_df.drop(['clusters'], axis=1, inplace=True)
kml3d3_df.drop(['clusters'], axis=1, inplace=True)
kml3d4_df.drop(['clusters'], axis=1, inplace=True)


# In[327]:


# kml3d1_df = kml3d1_df.set_index(['FIPS_y', 'TimePeriod'])
# kml3d2_df = kml3d2_df.set_index(['FIPS_y', 'TimePeriod'])
# kml3d3_df = kml3d3_df.set_index(['FIPS_y', 'TimePeriod'])
# kml3d4_df = kml3d4_df.set_index(['FIPS_y', 'TimePeriod'])


# In[328]:


# List of DataFrames
cluster_dataframes = [kml3d1_df, kml3d2_df, kml3d3_df, kml3d4_df]

p_value = 0.1
max_cond_vars = len(kml3d1_df.columns) - 2  # Assuming all DataFrames have the same columns

pp = PdfPages("DAGOutputs_cluster2_kml3d.pdf")


def graph_DAG(edges, df, title="Directed Acyclic Graph"):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.columns if key not in edge]
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
    color_map = ['C0' for g in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                     arrows=True, font_color='k', font_size=26, alpha=1, width=1,
                     edge_color='C1',
                     arrowstyle=ArrowStyle('Fancy, head_length=3, head_width=1.5, tail_width=.1'), ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches="tight")

for idx, c_df in enumerate(cluster_dataframes):
    c = PC(c_df)
    n = c_df.shape[0]
    model = c.estimate(return_type='pdag', variant='parallel', significance_level= p_value, max_cond_vars=max_cond_vars, ci_test='pearsonr')
    edges = model.edges

    # Call the graph_DAG function for each cluster DataFrame
    #graph_DAG(edges, c_df, title=f'Cluster {idx + 1} Directed Acyclic Graph')
    graph_DAG(edges, c_df, title=f'Cluster {idx + 1}\np_value = {p_value}\nnum_observations = {n}' )

pp.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[329]:


columns = ['Agriculture, forestry, fishing and hunting_x', 
                       'All industry total_x',
                       'Mining, quarrying, and oil and gas extraction_x',
                       'Utilities_x',
                      'NeighborGDP_y',
                      'M4',
                      'M4 Interest Rate']


# In[330]:


level_data = df1


# In[331]:


level_data = level_data[columns]


# In[332]:


level_data.rename(columns = {key:key[:4].replace("\n", "") for key in level_data.keys()}, inplace = True)
list(level_data.keys())


# In[333]:


level_data = level_data.dropna()


# In[334]:


rates_data = np.log(level_data).replace([np.inf, -np.inf], np.nan)


# In[335]:


diff_rates_data = rates_data.groupby('FIPS_y').diff()


# In[336]:


level_data.to_csv('level_data.csv', index=True)


# In[337]:


rates_data.to_csv('rates_data.csv', index=True)


# In[338]:


diff_rates_data.to_csv('diff_rates_data.csv', index=True)


# ## After performing the kml3d clustering in R for the three different data

# In[339]:


dfg_cl = level_data.copy()


# In[340]:


dfg_cl.reset_index(inplace=True)


# In[341]:


level = pd.read_csv('kml3d_cluster_Level_data.csv')


# In[342]:


merged_df = pd.merge(dfg_cl, level[['FIPS_y', 'clusters']], on='FIPS_y', how='left')


# In[343]:


kml3d_cl_level = merged_df.dropna()


# In[344]:


kml3d_cl_level = kml3d_cl_level.set_index(['FIPS_y', 'TimePeriod'])


# In[345]:


cluster_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}


# In[346]:


kml3d_cl_level['clusters'] = kml3d_cl_level['clusters'].map(cluster_mapping)


# In[347]:


features = kml3d_cl_level[['Agri', 'All ', 'Mini', 'Util', 'Neig', 'M4', 'M4 I']]


# In[348]:


kml3d_cl_data1 = kml3d_cl_level


# In[349]:


kml3d_cl_data = kml3d_cl_level


# In[350]:


# Create a dictionary to store the regression results
regression_results = {}

# Iterate through each cluster category
for cluster_category in kml3d_cl_data['clusters'].unique():
    # Create a dictionary to store the regression results for the current cluster category
    cluster_results = {}
    
    # Iterate through each variable
    for variable in features.columns:
        # Perform a regression for the current cluster category against the current variable
        X = features[[variable]]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = (kml3d_cl_data['clusters'] == cluster_category).astype(int)  # Binary outcome
        
        # Create and fit a logistic regression model
        model = sm.Logit(y, X).fit()
        
        # Store the regression coefficients in the dictionary
        cluster_results[variable] = {
            'Coefficient': model.params[variable],
            'Intercept': model.params['const'],
            'P-value': model.pvalues[variable]
        }
    
    # Store the cluster-specific regression results in the regression_results dictionary
    regression_results[cluster_category] = cluster_results

# Create a DataFrame to hold the regression coefficients for each cluster category and variable
heatmap_data = pd.DataFrame()

# Iterate through each cluster category and variable to construct the DataFrame
for cluster_category, cluster_results in regression_results.items():
    for variable, results in cluster_results.items():
        row_data = {
            'Cluster_Category': cluster_category,
            'Variable': variable,
            'Coefficient': results['Coefficient'],
            'P-value': results['P-value']
        }
        heatmap_data = heatmap_data.append(row_data, ignore_index=True)

# Pivot the DataFrame to prepare for the heatmap
heatmap_data_pivot = heatmap_data.pivot('Variable', 'Cluster_Category', 'Coefficient')

# Create a heatmap of the regression coefficients
plt.figure(figsize=(12, 8))
max_val = np.abs(heatmap_data_pivot).max().max()
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt=".4f", linewidths=0.5, vmin=max_val*-1, vmax=max_val)
plt.title('Logistic Regression Coefficients Heatmap between Cluster Categories and Variables')
plt.show()


# In[351]:


kml3d_cl_data = kml3d_cl_data.reset_index()


# In[352]:


unique_kml3d = kml3d_cl_data.drop_duplicates(subset='FIPS_y')


# In[353]:


unique_kml3d= unique_kml3d.rename(columns={'FIPS_y': 'fips_code'})


# In[354]:


kml3d_map = pd.merge(map_data, unique_kml3d, left_index=True, right_on='fips_code')


# In[355]:


hover_data = ["fips_code", "state", "NAME"]
fig = px.choropleth(kml3d_map, 
                    geojson=kml3d_map['geometry'],  
                    locations=kml3d_map.index, 
                    color="clusters",
                    scope="usa", hover_data=hover_data 
                   )

fig.update_geos(
    visible=False, 
    showcoastlines=True,  
)

fig.update_layout(
    title="Clustering of Counties",
    geo=dict(
        center={"lat": 37.0902, "lon": -95.7129},  
        projection_scale=6
    )
)

fig.show()

fig.write_html('kml3d_level.html')


# In[356]:


kml3d_cl_data = kml3d_cl_data.set_index(['FIPS_y', 'TimePeriod'])


# In[357]:


grouped = kml3d_cl_data.groupby('clusters')

# Initialize an empty dictionary to store the DataFrames for each cluster
cluster_dataframes = {}

# Iterate over the groups and create DataFrames
for cluster, group in grouped:
    cluster_dataframes[cluster] = group


# In[358]:


#Cluster dataframes
kml3d1_df = cluster_dataframes[1]
kml3d2_df = cluster_dataframes[2]
kml3d3_df = cluster_dataframes[3]
kml3d4_df = cluster_dataframes[4]


# In[359]:


kml3d1_df.drop(['clusters'], axis=1, inplace=True)
kml3d2_df.drop(['clusters'], axis=1, inplace=True)
kml3d3_df.drop(['clusters'], axis=1, inplace=True)
kml3d4_df.drop(['clusters'], axis=1, inplace=True)


# In[360]:


# List of DataFrames
cluster_dataframes = [kml3d1_df, kml3d2_df, kml3d3_df, kml3d4_df]

p_value = 0.1
max_cond_vars = len(kml3d1_df.columns) - 2  # Assuming all DataFrames have the same columns

pp = PdfPages("DAGOutputs_cluster2_kml3d_level.pdf")


def graph_DAG(edges, df, title="Directed Acyclic Graph"):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.columns if key not in edge]
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
    color_map = ['C0' for g in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                     arrows=True, font_color='k', font_size=26, alpha=1, width=1,
                     edge_color='C1',
                     arrowstyle=ArrowStyle('Fancy, head_length=3, head_width=1.5, tail_width=.1'), ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches="tight")

for idx, c_df in enumerate(cluster_dataframes):
    c = PC(c_df)
    n = c_df.shape[0]
    model = c.estimate(return_type='pdag', variant='parallel', significance_level= p_value, max_cond_vars=max_cond_vars, ci_test='pearsonr')
    edges = model.edges

    # Call the graph_DAG function for each cluster DataFrame
    #graph_DAG(edges, c_df, title=f'Cluster {idx + 1} Directed Acyclic Graph')
    graph_DAG(edges, c_df, title=f'Cluster {idx + 1}\np_value = {p_value}\nnum_observations = {n}' )

pp.close()


# ## For Rates

# In[361]:


dfg_cl = level_data.copy()


# In[362]:


dfg_cl.reset_index(inplace=True)


# In[363]:


level = pd.read_csv('kml3d_cluster_rates_data.csv')


# In[364]:


merged_df = pd.merge(dfg_cl, level[['FIPS_y', 'clusters']], on='FIPS_y', how='left')


# In[365]:


kml3d_cl_level = merged_df.dropna()


# In[366]:


kml3d_cl_level = kml3d_cl_level.set_index(['FIPS_y', 'TimePeriod'])


# In[367]:


kml3d_cl_level['clusters'] = kml3d_cl_level['clusters'].map(cluster_mapping)


# In[368]:


features = kml3d_cl_level[['Agri', 'All ', 'Mini', 'Util', 'Neig', 'M4', 'M4 I']]


# In[369]:


kml3d_cl_data2 = kml3d_cl_level


# In[370]:


kml3d_cl_data = kml3d_cl_level


# In[371]:


# Create a dictionary to store the regression results
regression_results = {}

# Iterate through each cluster category
for cluster_category in kml3d_cl_data['clusters'].unique():
    # Create a dictionary to store the regression results for the current cluster category
    cluster_results = {}
    
    # Iterate through each variable
    for variable in features.columns:
        # Perform a regression for the current cluster category against the current variable
        X = features[[variable]]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = (kml3d_cl_data['clusters'] == cluster_category).astype(int)  # Binary outcome
        
        # Create and fit a logistic regression model
        model = sm.Logit(y, X).fit()
        
        # Store the regression coefficients in the dictionary
        cluster_results[variable] = {
            'Coefficient': model.params[variable],
            'Intercept': model.params['const'],
            'P-value': model.pvalues[variable]
        }
    
    # Store the cluster-specific regression results in the regression_results dictionary
    regression_results[cluster_category] = cluster_results

# Create a DataFrame to hold the regression coefficients for each cluster category and variable
heatmap_data = pd.DataFrame()

# Iterate through each cluster category and variable to construct the DataFrame
for cluster_category, cluster_results in regression_results.items():
    for variable, results in cluster_results.items():
        row_data = {
            'Cluster_Category': cluster_category,
            'Variable': variable,
            'Coefficient': results['Coefficient'],
            'P-value': results['P-value']
        }
        heatmap_data = heatmap_data.append(row_data, ignore_index=True)

# Pivot the DataFrame to prepare for the heatmap
heatmap_data_pivot = heatmap_data.pivot('Variable', 'Cluster_Category', 'Coefficient')

# Create a heatmap of the regression coefficients
plt.figure(figsize=(12, 8))
max_val = np.abs(heatmap_data_pivot).max().max()
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt=".4f", linewidths=0.5, vmin=max_val*-1, vmax=max_val)
plt.title('Logistic Regression Coefficients Heatmap between Cluster Categories and Variables')
plt.show()


# In[372]:


kml3d_cl_data = kml3d_cl_data.reset_index()


# In[373]:


unique_kml3d = kml3d_cl_data.drop_duplicates(subset='FIPS_y')


# In[374]:


unique_kml3d= unique_kml3d.rename(columns={'FIPS_y': 'fips_code'})


# In[375]:


kml3d_map = pd.merge(map_data, unique_kml3d, left_index=True, right_on='fips_code')


# In[376]:


fig = px.choropleth(kml3d_map, 
                    geojson=kml3d_map['geometry'],  
                    locations=kml3d_map.index, 
                    color="clusters",
                    scope="usa", hover_data=hover_data 
                   )

fig.update_geos(
    visible=False, 
    showcoastlines=True,  
)

fig.update_layout(
    title="Clustering of Counties",
    geo=dict(
        center={"lat": 37.0902, "lon": -95.7129},  
        projection_scale=6
    )
)

fig.show()

fig.write_html('kml3d_rates.html')


# In[377]:


kml3d_cl_data = kml3d_cl_data.set_index(['FIPS_y', 'TimePeriod'])


# In[378]:


grouped = kml3d_cl_data.groupby('clusters')

# Initialize an empty dictionary to store the DataFrames for each cluster
cluster_dataframes = {}

# Iterate over the groups and create DataFrames
for cluster, group in grouped:
    cluster_dataframes[cluster] = group


# In[379]:


#Cluster dataframes
kml3d1_df = cluster_dataframes[1]
kml3d2_df = cluster_dataframes[2]
kml3d3_df = cluster_dataframes[3]
kml3d4_df = cluster_dataframes[4]


# In[380]:


kml3d1_df.drop(['clusters'], axis=1, inplace=True)
kml3d2_df.drop(['clusters'], axis=1, inplace=True)
kml3d3_df.drop(['clusters'], axis=1, inplace=True)
kml3d4_df.drop(['clusters'], axis=1, inplace=True)


# In[381]:


# List of DataFrames
cluster_dataframes = [kml3d1_df, kml3d2_df, kml3d3_df, kml3d4_df]

p_value = 0.1
max_cond_vars = len(kml3d1_df.columns) - 2  # Assuming all DataFrames have the same columns

pp = PdfPages("DAGOutputs_cluster2_kml3d_rates.pdf")


def graph_DAG(edges, df, title="Directed Acyclic Graph"):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.columns if key not in edge]
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
    color_map = ['C0' for g in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                     arrows=True, font_color='k', font_size=26, alpha=1, width=1,
                     edge_color='C1',
                     arrowstyle=ArrowStyle('Fancy, head_length=3, head_width=1.5, tail_width=.1'), ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches="tight")

for idx, c_df in enumerate(cluster_dataframes):
    c = PC(c_df)
    n = c_df.shape[0]
    model = c.estimate(return_type='pdag', variant='parallel', significance_level= p_value, max_cond_vars=max_cond_vars, ci_test='pearsonr')
    edges = model.edges

    # Call the graph_DAG function for each cluster DataFrame
    #graph_DAG(edges, c_df, title=f'Cluster {idx + 1} Directed Acyclic Graph')
    graph_DAG(edges, c_df, title=f'Cluster {idx + 1}\np_value = {p_value}\nnum_observations = {n}' )

pp.close()


# ## For Diff rates

# In[382]:


dfg_cl = level_data.copy()


# In[383]:


dfg_cl.reset_index(inplace=True)


# In[384]:


level = pd.read_csv('kml3d_cluster_diff_rates_data.csv')


# In[385]:


merged_df = pd.merge(dfg_cl, level[['FIPS_y', 'clusters']], on='FIPS_y', how='left')


# In[386]:


kml3d_cl_level = merged_df.dropna()


# In[387]:


kml3d_cl_level = kml3d_cl_level.set_index(['FIPS_y', 'TimePeriod'])


# In[388]:


kml3d_cl_level['clusters'] = kml3d_cl_level['clusters'].map(cluster_mapping)


# In[389]:


features = kml3d_cl_level[['Agri', 'All ', 'Mini', 'Util', 'Neig', 'M4', 'M4 I']]


# In[390]:


kml3d_cl_data3 = kml3d_cl_level


# In[391]:


kml3d_cl_data = kml3d_cl_level


# In[392]:


# Create a dictionary to store the regression results
regression_results = {}

# Iterate through each cluster category
for cluster_category in kml3d_cl_data['clusters'].unique():
    # Create a dictionary to store the regression results for the current cluster category
    cluster_results = {}
    
    # Iterate through each variable
    for variable in features.columns:
        # Perform a regression for the current cluster category against the current variable
        X = features[[variable]]
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = (kml3d_cl_data['clusters'] == cluster_category).astype(int)  # Binary outcome
        
        # Create and fit a logistic regression model
        model = sm.Logit(y, X).fit()
        
        # Store the regression coefficients in the dictionary
        cluster_results[variable] = {
            'Coefficient': model.params[variable],
            'Intercept': model.params['const'],
            'P-value': model.pvalues[variable]
        }
    
    # Store the cluster-specific regression results in the regression_results dictionary
    regression_results[cluster_category] = cluster_results

# Create a DataFrame to hold the regression coefficients for each cluster category and variable
heatmap_data = pd.DataFrame()

# Iterate through each cluster category and variable to construct the DataFrame
for cluster_category, cluster_results in regression_results.items():
    for variable, results in cluster_results.items():
        row_data = {
            'Cluster_Category': cluster_category,
            'Variable': variable,
            'Coefficient': results['Coefficient'],
            'P-value': results['P-value']
        }
        heatmap_data = heatmap_data.append(row_data, ignore_index=True)

# Pivot the DataFrame to prepare for the heatmap
heatmap_data_pivot = heatmap_data.pivot('Variable', 'Cluster_Category', 'Coefficient')

# Create a heatmap of the regression coefficients
plt.figure(figsize=(12, 8))
max_val = np.abs(heatmap_data_pivot).max().max()
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt=".4f", linewidths=0.5, vmin=max_val*-1, vmax=max_val)
plt.title('Logistic Regression Coefficients Heatmap between Cluster Categories and Variables')
plt.show()


# In[393]:


kml3d_cl_data = kml3d_cl_data.reset_index()


# In[394]:


unique_kml3d = kml3d_cl_data.drop_duplicates(subset='FIPS_y')


# In[395]:


unique_kml3d= unique_kml3d.rename(columns={'FIPS_y': 'fips_code'})


# In[396]:


kml3d_map = pd.merge(map_data, unique_kml3d, left_index=True, right_on='fips_code')


# In[397]:


fig = px.choropleth(kml3d_map, 
                    geojson=kml3d_map['geometry'],  
                    locations=kml3d_map.index, 
                    color="clusters",
                    scope="usa", hover_data=hover_data 
                   )

fig.update_geos(
    visible=False, 
    showcoastlines=True,  
)

fig.update_layout(
    title="Clustering of Counties",
    geo=dict(
        center={"lat": 37.0902, "lon": -95.7129},  
        projection_scale=6
    )
)

fig.show()

fig.write_html('kml3d_map_diff_rates.html')


# In[398]:


kml3d_cl_data = kml3d_cl_data.set_index(['FIPS_y', 'TimePeriod'])


# In[399]:


grouped = kml3d_cl_data.groupby('clusters')

# Initialize an empty dictionary to store the DataFrames for each cluster
cluster_dataframes = {}

# Iterate over the groups and create DataFrames
for cluster, group in grouped:
    cluster_dataframes[cluster] = group


# In[400]:


#Cluster dataframes
kml3d1_df = cluster_dataframes[1]
kml3d2_df = cluster_dataframes[2]
kml3d3_df = cluster_dataframes[3]
kml3d4_df = cluster_dataframes[4]


# In[401]:


kml3d1_df.drop(['clusters'], axis=1, inplace=True)
kml3d2_df.drop(['clusters'], axis=1, inplace=True)
kml3d3_df.drop(['clusters'], axis=1, inplace=True)
kml3d4_df.drop(['clusters'], axis=1, inplace=True)


# In[402]:


# List of DataFrames
cluster_dataframes = [kml3d1_df, kml3d2_df, kml3d3_df, kml3d4_df]

p_value = 0.1
max_cond_vars = len(kml3d1_df.columns) - 2  # Assuming all DataFrames have the same columns

pp = PdfPages("DAGOutputs_cluster2_kml3d_diif_rates.pdf")


def graph_DAG(edges, df, title="Directed Acyclic Graph"):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.columns if key not in edge]
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
    color_map = ['C0' for g in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                     arrows=True, font_color='k', font_size=26, alpha=1, width=1,
                     edge_color='C1',
                     arrowstyle=ArrowStyle('Fancy, head_length=3, head_width=1.5, tail_width=.1'), ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches="tight")

for idx, c_df in enumerate(cluster_dataframes):
    c = PC(c_df)
    n = c_df.shape[0]
    model = c.estimate(return_type='pdag', variant='parallel', significance_level= p_value, max_cond_vars=max_cond_vars, ci_test='pearsonr')
    edges = model.edges

    # Call the graph_DAG function for each cluster DataFrame
    #graph_DAG(edges, c_df, title=f'Cluster {idx + 1} Directed Acyclic Graph')
    graph_DAG(edges, c_df, title=f'Cluster {idx + 1}\np_value = {p_value}\nnum_observations = {n}' )

pp.close()


# In[ ]:





# In[ ]:





# In[403]:


from PIL import Image


# In[404]:


from matplotlib.backends.backend_pdf import PdfPages
from pgmpy.estimators import PC
import matplotlib.pyplot as plt
import networkx as nx

# List of your three different datasets (replace these with your actual datasets)
dataframes = [kml3d_cl_data1, kml3d_cl_data2, kml3d_cl_data3]  # Replace with your actual DataFrames

# List of p-values for each dataframe
p_values_list = [[0.1, 0.05, 0.01], [0.1, 0.05, 0.01], [0.1, 0.05, 0.01]]  # Replace with your p-values

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

def graph_DAG(edges, df, title="Directed Acyclic Graph", pp=None):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.columns if key not in edge]
        controls = list(set(controls))
        keep_controls = []
        for control in controls:
            control_edges = [ctrl_edge for ctrl_edge in edges if control == ctrl_edge[0]]
            if (control, edge[1]) in control_edges:
                keep_controls.append(control)
        pcorr = df[[edge[0], edge[1]] + keep_controls].pcorr()
        edge_labels[edge] = str(round(pcorr[edge[0]].loc[edge[1]], 2))
    graph.add_edges_from(edges)
    color_map = ['C0' for g in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                     arrows=True, font_color='k', font_size=26, alpha=1, width=1,
                     edge_color='C1', ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches="tight")

# Iterate over each dataframe
for idx, dataframe in enumerate(dataframes):
    # Get clustered dataframes for the current dataframe
    clustered_dfs = create_cluster_dags(dataframe)
    
    # List of p-values for the current dataframe
    p_values = p_values_list[idx]

    # Create a PDF file to save multiple DAGs for the current dataframe
    pdf_file = f"DAGOutputs_dataframe_{idx + 1}.pdf"
    pp = PdfPages(pdf_file)

    # Iterate over clustered dataframes and p-values to create DAGs
    for cluster_idx, cluster_df in enumerate(clustered_dfs):
        for p_value in p_values:
            c = PC(cluster_df)
            model = c.estimate(return_type='pdag', variant='parallel', significance_level=p_value, ci_test='pearsonr')
            edges = model.edges

            # Create DAG for the current cluster and p-value using the provided graph_DAG function
            graph_DAG(edges, cluster_df, title=f'Dataframe {idx + 1} - p_value = {p_value}, Cluster {cluster_idx + 1}', pp=pp)

    pp.close()


# ## DAGs for various levels of data and different p_values

# In[405]:


from matplotlib.backends.backend_pdf import PdfPages
from pgmpy.estimators import PC
import matplotlib.pyplot as plt
import networkx as nx

# List of your three different datasets (replace these with your actual datasets)
dataframes = [kml3d_cl_data1, kml3d_cl_data2, kml3d_cl_data3]  # Replace with your actual DataFrames

# List of p-values for each dataframe
p_values_list = [[0.1, 0.05, 0.01], [0.1, 0.05, 0.01], [0.1, 0.05, 0.01]]  # Replace with your p-values

# Define names for the output files based on dataframes
output_names = ['level_data', 'rates_data', 'diff_rates_data']

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

def graph_DAG(edges, df, title="Directed Acyclic Graph", pp=None):
    graph = nx.DiGraph()
    edge_labels = {}
    for edge in edges:
        controls = [key for key in df.columns if key not in edge]
        controls = list(set(controls))
        keep_controls = []
        for control in controls:
            control_edges = [ctrl_edge for ctrl_edge in edges if control == ctrl_edge[0]]
            if (control, edge[1]) in control_edges:
                keep_controls.append(control)
        pcorr = df[[edge[0], edge[1]] + keep_controls].pcorr()
        edge_labels[edge] = str(round(pcorr[edge[0]].loc[edge[1]], 2))
    graph.add_edges_from(edges)
    color_map = ['C0' for g in graph]

    fig, ax = plt.subplots(figsize=(20, 12))
    graph.nodes()
    plt.tight_layout()
    pos = nx.spring_layout(graph)
    
    plt.title(title, fontsize=30)
    nx.draw_networkx(graph, pos, node_color=color_map, node_size=1200, with_labels=True,
                     arrows=True, font_color='k', font_size=26, alpha=1, width=1,
                     edge_color='C1', ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green', font_size=20)
    pp.savefig(fig, bbox_inches="tight")

# Iterate over each dataframe
for idx, dataframe in enumerate(dataframes):
    # Get clustered dataframes for the current dataframe
    clustered_dfs = create_cluster_dags(dataframe)
    
    # List of p-values for the current dataframe
    p_values = p_values_list[idx]

    # Create a PDF file to save multiple DAGs for the current dataframe
    pdf_file = f"DAGOutputs_{output_names[idx]}.pdf"
    pp = PdfPages(pdf_file)

    # Iterate over clustered dataframes and p-values to create DAGs
    for cluster_idx, cluster_df in enumerate(clustered_dfs):
        for p_value in p_values:
            c = PC(cluster_df)
            model = c.estimate(return_type='pdag', variant='parallel', significance_level=p_value, ci_test='pearsonr')
            edges = model.edges

            # Create DAG for the current cluster and p-value using the provided graph_DAG function
            graph_DAG(edges, cluster_df, title=f'Dataframe {idx + 1} - p_value = {p_value}, Cluster {cluster_idx + 1}', pp=pp)

    pp.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




