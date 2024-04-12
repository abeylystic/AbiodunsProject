
library(rgl)

#install.packages('misc3d')

library(dplyr)
library(tidyr)
library(kml3d)
library(kml)

#setwd("/Users/abiodun.idowu/Library/CloudStorage/OneDrive-NorthDakotaUniversitySystem/Desktop/PhD/BEA project/notebook_to_start")

data <- read.csv("cluster_data.csv")

#complete_combinations <- expand.grid(FIPS_y = unique(data$FIPS_y), 
#                                     TimePeriod = unique(data$TimePeriod))

# Step 2: Create a new data frame with complete combinations
#new_data <- complete_combinations

# Step 3: Left join your original data to the new data frame, filling in missing values with NA
#final_data <- new_data %>%
#  left_join(data, by = c("FIPS_y", "TimePeriod"))

# Replace NAs with a specific value if needed
#final_data[is.na(final_data)] <- 0  # Replace NAs with 0 (you can choose a different value if needed)

#data1 <- final_data

# Pivot the data from wide to long format
df_cl <- pivot_wider(
  data = data,
  id_cols = FIPS_y,
  names_from = TimePeriod,
  values_from = -FIPS_y,  # Select all columns starting from the third column
  names_prefix = ""
)

df_cl <- df_cl[, -c(2:19)]

df_cl1 <- na.omit(df_cl)

idAll <- as.character(unique(df_cl1$FIPS_y))

cld_data <- clusterLongData3d(df_cl1, idAll = idAll, timeInData= list(x=c(2:19),
                                                                      y=c(20:37),
                                                                      z=c(38:55),
                                                                      a=c(56:73),
                                                                      b=c(74:91),
                                                                      c=c(92:109),
                                                                      d=c(110:127),
                                                                      e=c(128:145)),
                              varNames= c("Agri", "Mini", "Util", "All", 
                                          "Neig", "GDP_", "M4", "M4I"))



# Close any existing graphics device
dev.off()

x11(width = 10, height = 6)

kml3d(cld_data, 2:8, nbRedrawing = 20, toPlot = 'criterion')

# Get clusters and store them in dataframe
df_cl1$clusters <- getClusters(cld_data, nbCluster = 4)

# Export the data
write.csv(df_cl1, file='kml3d_cluster_data.csv', row.names=FALSE)


try(choice(cld_data))

x11(width = 10, height = 6)

plotMeans3d(cld_data, 2,3,4,5,6,7,8)
