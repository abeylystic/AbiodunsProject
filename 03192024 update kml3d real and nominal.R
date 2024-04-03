setwd("/Users/abiodun.idowu/Library/CloudStorage/OneDrive-NorthDakotaUniversitySystem/Desktop/PhD/BEA project/notebook_to_start")

library(rgl)
library(dplyr)
library(tidyr)
library(kml3d)
library(kml)

# ai_data = rates, start with all variables
##################################################################################################################
data <- read.csv("nominal.csv")
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
                                                                     c=c(102:121)),
                              varNames= c("Agri","Mini", "Util", 
                                          "M4_y", "M4I", "unem"))

kml3d(cld_data, 2:6, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl$clusters <- getClusters(cld_data, nbCluster = 4)

# Export the data
write.csv(df_cl, file='nominal_kml3d.csv', row.names=FALSE)


# Next remove total gdp
################################################################################
data2 <- read.csv("diff_nominal.csv")

#data2 <- subset(data, select = -c(All))

complete_combinations <- expand.grid(FIPS_y = unique(data2$FIPS_y), 
                                     TimePeriod = unique(data2$TimePeriod))

# Step 2: Create a new data frame with complete combinations
new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
final_data <- new_data %>%
  left_join(data2, by = c("FIPS_y", "TimePeriod"))

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
                                                                     c=c(102:121)),
                              varNames= c("Agri","Mini", "Util",
                                          "M4_y", "M4I", "unem"))

kml3d(cld_data, 2:6, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl$clusters <- getClusters(cld_data, nbCluster = 4)

# Export the data
write.csv(df_cl, file='diff_nominal_kml3d.csv', row.names=FALSE)


##############################################################################

# Next remove total gdp and neighbors gdp
################################################################################
data3 <- read.csv("real.csv")

#data2 <- subset(data, select = -c(All))

complete_combinations <- expand.grid(FIPS_y = unique(data3$FIPS_y), 
                                     TimePeriod = unique(data3$TimePeriod))

# Step 2: Create a new data frame with complete combinations
new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
final_data <- new_data %>%
  left_join(data3, by = c("FIPS_y", "TimePeriod"))

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
                                                                     c=c(102:121)),
                              varNames= c("Agri","Mini", "Util",
                                          "M4_y", "M4I", "unem"))

kml3d(cld_data, 2:6, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl$clusters <- getClusters(cld_data, nbCluster = 4)

# Export the data
write.csv(df_cl, file='real_kml3d.csv', row.names=FALSE)

####################################################################################


#####################################################################################

data4 <- read.csv("diff_real.csv")

complete_combinations <- expand.grid(FIPS_y = unique(data4$FIPS_y), 
                                     TimePeriod = unique(data4$TimePeriod))

# Step 2: Create a new data frame with complete combinations
new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
final_data <- new_data %>%
  left_join(data4, by = c("FIPS_y", "TimePeriod"))

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
                                                                     c=c(102:121)),
                              varNames= c("Agri","Mini", "Util",
                                          "M4_y", "M4I", "unem"))

kml3d(cld_data, 2:6, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl$clusters <- getClusters(cld_data, nbCluster = 4)

# Export the data
write.csv(df_cl, file='diff_real_kml3d.csv', row.names=FALSE)
