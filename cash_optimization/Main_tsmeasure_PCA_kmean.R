library(dplyr)
library(tsfeatures)
library(tidyverse)
library('anomalousACM')
library(mclust)
library("factoextra")
library("magrittr")
library("NbClust")

#tsfeature
library(tsfeatures)
library(tidyverse)
set.seed(80)

ts_features_preprocessing <- function(data){
  rownames(data) <- data[,1]
  data <- data[,-1]
  #data <- na.omit(data)
  data <- scale(data)
  data <- data[, colSums(is.na(data)) != nrow(data)]
  data <- replace(data, is.na(data), 0)
  return (data)
}

dimension_reduction <- function(data, pca=TRUE, tsne=FALSE){
  
  if (tsne){
    ## Rtsne function may take some minutes to complete...
    data = Rtsne(as.matrix(data), perplexity=2, pca=FALSE)
    ## getting the two dimension matrix
    data = as.data.frame(data$Y)
  }
  
  if (pca){
    pca  <- prcomp(data, retx=T, scale.=T) # scaled pca [exclude species col]
    data <- pca$x[,1:3]  
  }
  
  return (data)
}

train_test_split <- function(data, split_ratio = 1){
  #Train and Test split
  smp_size <- floor(split_ratio * nrow(data))
  
  ## set the seed to make your partition reproducible
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  train <- data[train_ind, ]
  test  <- data[-train_ind,]
  
  return (list(train=train, test=test))
}

optimal_clusters_num <- function(data){
  nb <- NbClust(data, distance = "euclidean", min.nc = 2,max.nc = floor(dim(data)[1]/2), method = "kmeans", index = 'hartigan')
  nbclusters = nb$Best.nc[1]
}

visualize_clusters <- function(train, clusters){
  #Clustering
  km.res <- kmeans(train, clusters, nstart = 25)
  
  # Visualize
  clus = fviz_cluster(km.res, data = train,
                      ellipse.type = "convex",
                      palette = "jco",
                      ggtheme = theme_minimal())
  plot(clus)
  
  #Interprating the clusters
  clusters = clus[1]$data
  X <- split(clusters, clusters$cluster)
  
  return (X)
}


#Feature extractor from time series
ts_measure_extrator <- function(similar_ts, ts_freq){
  #similar_ts <- t(similar_ts)
  #Extracting feature vector using the tsmeasures() in tsfeature package.
  feature_mat = NULL
  for(k in 1 :ncol(similar_ts)){
    input<-ts(similar_ts[,k] ,freq=ts_freq)
    y <- tsmeasures(input)
    feature_mat <- rbind(feature_mat,y)
  }
  rownames(feature_mat) = colnames(similar_ts)
  #feature_mat[is.na(feature_mat)] <- 0
  write.csv(data.frame(feature_mat),file = "extracted_ts_measure.csv",row.names = TRUE,col.names = TRUE)
  features = read.csv("extracted_ts_measure.csv", header = TRUE, sep=',')
  return(features)
}

ts_distinct_data <- function(data_all,freq){
  unique_freq <- freq
  freq_col    <- data_all$V2
  similar_ts  <- data.frame(t(data_all))[freq_col == unique_freq]
  similar_ts  <- similar_ts[-1, ] 
  return(similar_ts)
}



data_preprocessing <- function(csv_file){
  csv_file_ = csv_file
  csv_file <- t(csv_file)
  csv_file <- csv_file[-c(3), ]
  colnames(csv_file) = csv_file[1, ] # the first row will be the header
  final_csv_file <- csv_file[-1, ] 
  
  #converting matrix to numeric class
  apply(final_csv_file, 2, as.numeric)
  sapply(final_csv_file, as.numeric)
  class(final_csv_file) <- "numeric"
  storage.mode(final_csv_file) <- "numeric"
  
  #converting to dataframe
  final_csv_file <- data.frame(t(final_csv_file))
  
  row.names(final_csv_file) <- data.frame(csv_file_)$V1
  return (final_csv_file)
}

#Reading Dataset of TS
csv_file_ <- read.csv("cif-dataset.txt", sep=";", header = FALSE)
data_all = data_preprocessing(csv_file_)
total_freq = distinct(select(data_all, 'V2'))

#spliting dataframe on basis of frequencies
for(i in 1:(ncol(total_freq)+1)){ 
  similar_ts <- ts_distinct_data(data_all,total_freq[i,1])
  extracted_ts_measure <- ts_measure_extrator(similar_ts, total_freq[i,1])
  data  <- ts_features_preprocessing(extracted_ts_measure)
  data  = dimension_reduction(data, pca=TRUE, tsne = FALSE)
  data  = train_test_split(data)
  train = data$train
  test  = data$train
  clusters_ = visualize_clusters(train, optimal_clusters_num(train))
  for (num in 0:length(clusters_)){
    print(clusters_[num])
  }
  
}


