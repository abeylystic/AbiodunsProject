setwd("/Users/abiodun.idowu/Library/CloudStorage/OneDrive-NorthDakotaUniversitySystem/Desktop/PhD/BEA project/notebook_to_start")

library(rgl)
library(dplyr)
library(tidyr)
library(kml3d)
library(kml)


##################################################################################################################
data <- read.csv("diff_rates_data.csv")
# Complete variables

complete_combinations <- expand.grid(FIPS_y = unique(data$FIPS_y), 
                                     TimePeriod = unique(data$TimePeriod))

# Step 2: Create a new data frame with complete combinations
new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
final_data <- new_data %>%
  left_join(data, by = c("FIPS_y", "TimePeriod"))

data1 <- final_data
#data1 <- na.omit(final_data)

# Pivot the data from wide to long format
df_cl <- pivot_wider(
  data = data1,
  id_cols = FIPS_y,
  names_from = TimePeriod,
  values_from = -FIPS_y,  # Select all columns starting from the third column
  names_prefix = ""
)

df_cl <- df_cl[, -c(2:21)]

idAll <- as.character(unique(df_cl$FIPS_y))

cld_data <- clusterLongData3d(df_cl, idAll = idAll, timeInData= list(x=c(2:21),
                                                                     y=c(22:41),
                                                                     z=c(42:61),
                                                                     a=c(62:81),
                                                                     b=c(82:101),
                                                                     c=c(102:121),
                                                                     d=c(122:141)),
                              varNames= c("Agri", "All", "Mini", "Util", 
                                          "Neig", "M4", "M4I"))

kml3d(cld_data, 2:8, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl$clusters <- getClusters(cld_data, nbCluster = 4)

# Export the data
write.csv(df_cl, file='kml3d_cluster_diff_data.csv', row.names=FALSE)

###############################################################################

# Remove M4, M4I and total gdp

names(data)

# drop total GDP
data_no_gdp_m4 <- subset(data, select = -c(All, M4, M4.I))

complete_combinations <- expand.grid(FIPS_y = unique(data_no_gdp_m4$FIPS_y), 
                                     TimePeriod = unique(data_no_gdp_m4$TimePeriod))

# Step 2: Create a new data frame with complete combinations
new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
final_data <- new_data %>%
  left_join(data_no_gdp_m4, by = c("FIPS_y", "TimePeriod"))

data1 <- final_data
#data1 <- na.omit(final_data)

# Pivot the data from wide to long format
df_cl <- pivot_wider(
  data = data1,
  id_cols = FIPS_y,
  names_from = TimePeriod,
  values_from = -FIPS_y,  # Select all columns starting from the third column
  names_prefix = ""
)

df_cl2 <- df_cl[, -c(2:21)]

idAll <- as.character(unique(df_cl2$FIPS_y))

cld_data <- clusterLongData3d(df_cl2, idAll = idAll, timeInData= list(x=c(2:21),
                                                                      y=c(22:41),
                                                                      z=c(42:61),
                                                                      a=c(62:81)),
                              varNames= c("Agri", "Mini", "Util", 
                                          "Neig"))

kml3d(cld_data, 2:8, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl2$clusters <- getClusters(cld_data, nbCluster = 4)

# warnings()

# Export the data
write.csv(df_cl2, file='kml3d_no_gdp_m4_diff.csv', row.names=FALSE)


###############################################################################

# Remove M4 and M4I

names(data)

# drop total GDP
data_no_m4 <- subset(data, select = -c(M4, M4.I))

complete_combinations <- expand.grid(FIPS_y = unique(data_no_m4$FIPS_y), 
                                     TimePeriod = unique(data_no_m4$TimePeriod))

# Step 2: Create a new data frame with complete combinations
new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
final_data <- new_data %>%
  left_join(data_no_m4, by = c("FIPS_y", "TimePeriod"))

data1 <- final_data
#data1 <- na.omit(final_data)

# Pivot the data from wide to long format
df_cl <- pivot_wider(
  data = data1,
  id_cols = FIPS_y,
  names_from = TimePeriod,
  values_from = -FIPS_y,  # Select all columns starting from the third column
  names_prefix = ""
)

df_cl2 <- df_cl[, -c(2:21)]

idAll <- as.character(unique(df_cl2$FIPS_y))

cld_data <- clusterLongData3d(df_cl, idAll = idAll, timeInData= list(x=c(2:21),
                                                                     y=c(22:41),
                                                                     z=c(42:61),
                                                                     a=c(62:81),
                                                                     b=c(82:101)),
                              varNames= c("Agri", "All", "Mini", "Util", 
                                          "Neig"))

kml3d(cld_data, 2:8, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl2$clusters <- getClusters(cld_data, nbCluster = 4)

# warnings()

# Export the data
write.csv(df_cl2, file='kml3d_no_m4_diff.csv', row.names=FALSE)

#################################################################################

# Remove neighbors gdp

names(data)

# drop
data_no_neigh <- subset(data, select = -c(Neig))

complete_combinations <- expand.grid(FIPS_y = unique(data_no_neigh$FIPS_y), 
                                     TimePeriod = unique(data_no_neigh$TimePeriod))

# Step 2: Create a new data frame with complete combinations
new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
final_data <- new_data %>%
  left_join(data_no_neigh, by = c("FIPS_y", "TimePeriod"))

data1 <- final_data
#data1 <- na.omit(final_data)

# Pivot the data from wide to long format
df_cl <- pivot_wider(
  data = data1,
  id_cols = FIPS_y,
  names_from = TimePeriod,
  values_from = -FIPS_y,  # Select all columns starting from the third column
  names_prefix = ""
)

df_cl2 <- df_cl[, -c(2:21)]

idAll <- as.character(unique(df_cl2$FIPS_y))

cld_data <- clusterLongData3d(df_cl2, idAll = idAll, timeInData= list(x=c(2:21),
                                                                      y=c(22:41),
                                                                      z=c(42:61),
                                                                      a=c(62:81),
                                                                      b=c(82:101),
                                                                      c=c(102:121)),
                              varNames= c("Agri", "All", "Mini", "Util",
                                          "M4", "M4I"))

kml3d(cld_data, 2:8, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl2$clusters <- getClusters(cld_data, nbCluster = 4)

# warnings()

# Export the data
write.csv(df_cl2, file='kml3d_no_neigh.csv', row.names=FALSE)


