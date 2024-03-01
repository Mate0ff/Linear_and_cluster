install.packages('factoextra')
library(factoextra)
install.packages('cluster')
library(cluster)
install.packages("readxl")
library("readxl")



data_km <- read_excel("...\Linear_and_cluster\sad_data.xlsx")
data_km <- data.frame(data_km)
names <- data_km$Country

df <- subset(data_km, select = -Country)
rownames(df) <- names
df <- subset(df, select = -Emissions_of_co2 )


df <- scale(df)

summary(df)



fviz_nbclust(df, pam, method = "wss")


set.seed(3)
kmed <- pam(df, k = 4)
fviz_cluster(kmed, data = df)
