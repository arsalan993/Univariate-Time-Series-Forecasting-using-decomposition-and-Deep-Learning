library(dplyr)
library(tsfeatures)
library(tidyverse)
library('anomalousACM')
library(mclust)
library("factoextra")
library("magrittr")
library("NbClust")
library(Rtsne)
library(anomalous)
library(stringr)
library(forecast)
library("keras") 
library("forecTheta")
library("cclust")

graphics.off()

k <- backend()
use_condaenv("r-tensorflow")
set.seed(0)

############ Batching the Dataset ############
batch_data <- function(train, labels, batch_size, out_w){
  #input window size
  inp_w = floor(out_w*(1.25))

  
  ##############################
  x_train = train
  y_train = labels
  
  if (batch_size > 0){
    last_index <- as.integer((dim(y_train)[1] / batch_size)) * batch_size
    index_diff <- (dim(y_train)[1] - last_index)
    if (index_diff > 0){
      index_diff <- batch_size - index_diff
      for (i in 1:index_diff){
        x_train <- rbind( as.vector(matrix(0,1,inp_w)),x_train)
        y_train <- rbind( as.vector(matrix(0,1,out_w)),y_train)
     }
    }
  }

  return (list(train_x=x_train, train_y=y_train))
}

train_validation_sets <- function(x_train, y_train, out_w){
  #input window size
  inp_w = floor(out_w*(1.25))
    
  #get validation set
  complete_data = cbind(x_train, y_train)
  #print("**************")
  #print(dim(complete_data))
  complete_data = train_test_split(complete_data, 0.1)
  train_data    = complete_data$train
  val_data      = complete_data$validation
  
  train  = train_data[1:inp_w]
  labels = train_data[(inp_w+1):(inp_w+out_w)]
  
  validation_x = val_data[1:inp_w]
  validation_y = val_data[(inp_w+1):(inp_w+out_w)]
    
  #cat(dim(train), dim(labels), "\n")
  #cat(dim(validation_x), dim(validation_y), "\n")
  #print("**************")
  
  return(list(train_x=train, train_y=labels, val_x=validation_x, val_y=validation_y))
}

############ Machine Learning Algo ############
LSTM <- function (train_x,train_y,test_x, test_y, batch_size,f_name, freq){
  file_name <- paste(f_name, freq, ".h5", sep="")
  train_x  <- k$eval(k_expand_dims(data.matrix(train_x), axis = 3))
  train_y <- data.matrix(train_y)
  
  cat("Training Data: ", dim(train_x), "\n")
  cat("Testing Data: ",  dim(train_y), "\n")
  
  
  input_shape <- c(dim(train_x)[1], dim(train_x)[2],dim(train_x)[3])
  n           <- input_shape[1]
  t_steps     <- input_shape[2]
  features    <- input_shape[3]
  epochs      <- 5000
  units       <- 100
  state       <- FALSE
  shuffle     <- FALSE
  
  cat(n, t_steps, features, "\n")
  
  #Test Samples
  test_x = k$eval(k_expand_dims(data.matrix(test_x), axis = 3))
  test_y = data.matrix(test_y)
  
  cat('Creating model:\n')
  model <- keras_model_sequential()
  
  model %>%
    layer_lstm(units = units, 
               batch_input_shape = c(batch_size,t_steps, features), 
               stateful = state,
               return_sequences = FALSE) %>%
    layer_gaussian_noise(0.005) %>% 
    # layer_dropout(rate = 0.09) %>% 
    # 
    # layer_lstm(units = 10,
    #            batch_input_shape = c(batch_size,t_steps, features),
    #            return_sequences = FALSE,
    #            activation = 'linear') %>%
    # layer_gaussian_noise(0.005) %>% 
    # layer_dropout(rate = 0.09) %>% 
    
    layer_dense(units = dim(train_y)[2], 
                use_bias=FALSE, 
                bias_initializer='zeros')
  
  # *** For retraining models
  if (file.exists(file_name)) {
    print("***************************")
    print("Loading Pre-Trained Weights")
    old_model <- load_model_hdf5(file_name, custom_objects = NULL, compile = TRUE)
    # copy weights
    old_weights = get_weights(old_model)
    set_weights(model, old_weights)
  }
  
  optimization_algo = 'adam' #adam, nadam, Adamax
  model %>% compile(loss = 'mae', optimizer = optimization_algo) 
  summary(model)
  
  cat('Training\n')
  par(mfrow=c(2,1))
  
  smape_score <- list()
  # for (i in 1:epochs){
  model %>% fit(x          = train_x,
                y          = train_y,
                batch_size = batch_size,
                epochs     = epochs,
                verbose    = 1,
                shuffle    = shuffle,
                validation_data = list(test_x, test_y),
                callbacks = list(
                  callback_early_stopping(monitor = 'val_loss',patience=5),
                  callback_reduce_lr_on_plateau(patience=3),
                  callback_model_checkpoint(file_name))
                )
  # model %>% reset_states()
  # }
  cat('Testing\n')
  
  #Model Prediction
  pred_out <- model %>%
    predict(test_x, batch_size = batch_size)
  
  #Calculation of Error
  error = errorMetric(c(t(test_y)), c(t(pred_out)), type = "sAPE", statistic = "M")
  smape_score[i] = error
  
  #Plot Graphs
  plot(c(t(pred_out)),type="l", col="red", pch = 1, xlab="Days", ylab="Amount")
  lines(c(t(test_y)),col="green")
  

  error = c(unlist(smape_score, use.names=FALSE))
  plot(error, type="b", col="black", pch = 2, xlab="Epochs", ylab= paste0("SMAPE Error ",f_name))
  cat("\n *********************************** \n Epoch: ", i, "\n \n")
  
  ##################### Generating the model for online forecasting #######################
  batch_size  <- 1
  cat('Creating model:\n')
  new_model <- keras_model_sequential()
  
  new_model %>%
    layer_lstm(units = units, batch_input_shape = c(batch_size,t_steps, features), stateful = state) %>%
    layer_dense(units = dim(train_y)[2], use_bias=FALSE, bias_initializer='zeros')
  
  # copy weights
  old_weights = get_weights(model)
  set_weights(new_model, old_weights)
  
  new_model %>% compile(loss = 'mae', optimizer = optimization_algo)
  summary(new_model)
  
  ######################################## FORECASTING ##############################################
  
  cat('Testing\n')
  
  #Model Prediction
  pred_out <- new_model %>%
    predict(test_x, batch_size = batch_size)
  
  #saving model
  save_model_hdf5(new_model, file_name, overwrite = TRUE, include_optimizer = TRUE)
  
  #Calculation of Error
  error = errorMetric(c(test_y), c(pred_out), type = "sAPE", statistic = "M")
  #print(error)
}


normalization <- function(X,Y=NULL,norm = TRUE, normalizer = NULL){
  inp_win = dim(X)[2]
  if (is.null(Y) == FALSE){
    out_win = dim(Y)[2]
    cmb <- cbind(X,Y)
  }else{
    cmb <- rbind(X,Y)
  }
  
  if(norm){
    normalizer = cmb[,inp_win]
    norm_cmb <- cmb - normalizer
  }
  else{
    norm_cmb <- cmb + normalizer
  }
  ind = inp_win+1
  
  if (is.null(Y)){
    
    return(list(X=norm_cmb, norm_vector=normalizer))
  }else {
    X_ = data.frame(norm_cmb[,1:inp_win])
    Y_ = data.frame(norm_cmb[,ind : dim(cmb)[2]])
    
    return(list(X=X_, Y=Y_))
  }
  
}


framing_dataset <- function(deseasonal_, freq){
  framing_container = NULL
  for (clust_ in 1:length(deseasonal_)){
    clust <- deseasonal_[clust_]
    clust <- data.frame(clust)
    #print(clust)
    framing_container[[clust_]] <- data_framing(clust, freq)
    
  }
  return(framing_container)
}


############################################
data_framing <- function(dataset, output_window){
  out_win = output_window
  inp_win = floor(output_window*1.25)
  
  trend_train <- dataset
  trend_train = data.matrix(trend_train)
  end_index = length(trend_train[,1])-inp_win-out_win+1
  start_y = inp_win+1
  m1 <- matrix(1:length(trend_train),nrow = dim(trend_train)[1],
               ncol = dim(trend_train)[2])[1:end_index,]
  X <- t(sapply(m1, function(x) trend_train[x:(x + inp_win -1)]))
  
  m2 <- matrix(1:length(trend_train)+inp_win,nrow = dim(trend_train)[1],
               ncol = dim(trend_train)[2])[1:end_index,]
  
  Y <- t(sapply(m2,function(x) trend_train[x:(x + out_win - 1)]))
  temp_XY <- cbind(X,Y)
  temp_XY <- na.omit(temp_XY)
  X <- temp_XY[,1:inp_win]
  Y <- temp_XY[,(inp_win+1):dim(temp_XY)[2]]

  norm_data<- normalization(X,Y)
  return(list(X = X, Y = Y, norm_X = norm_data$X ,norm_Y = norm_data$Y))
  
}
###############################
deseasonalization <- function(x, columns, freq, prediction = FALSE) {
  n = length(columns)
  
  if((length(x)%%n)==0) {
    x = t(matrix(x, nrow=n))
    colnames(x) = columns
  } 
  else {
    x = t(matrix(append(x, rep(NA, n-(length(x)%%n))), nrow=n))
    colnames(x) = columns
  } 
  
  # time_series = ts( x[,"trend"] + x[,"remainder"]+ x[,"seasonal"],  frequency = freq)
  # deseason    = ts( x[,"trend"] + x[,"remainder"],  frequency = freq)
  # trend       = ts(x[,"trend"],  frequency = freq)
  # remainder   = ts(x[,"remainder"],  frequency = freq)
  # seasonal    = ts(x[,"seasonal"],  frequency = freq)
  # 
  # time_series = cbind(time_series, deseason, trend, remainder, seasonal)
  # plot(time_series)
  
  if(prediction){
    return(x[,"seasonal"])
  }else{
    return(x[,"trend"] + x[,"remainder"])
    #return(x[,"trend"] + x[,"remainder"]+x[,"seasonal"])
  }
}

remove_season <- function(stl_output_, freq){
  all_deseasonal_data = NULL
  
  for (nclus in seq_along(stl_output_)){
    stl_output = stl_output_[[nclus]]
    
    deseasonal_data = NULL
    column_names = colnames(stl_output)
    #print(column_names)
    for(i in 1:length(column_names)){
      tseries  = stl_output[,column_names[i]]
      deseason_ts = deseasonalization(tseries, c('seasonal', 'trend', 'remainder'), freq)
      deseasonal_data = cbind(deseasonal_data, as.vector(t(deseason_ts)))
      colnames(deseasonal_data)[i] = column_names[i]
    }
    
    all_deseasonal_data[[nclus]] <- deseasonal_data
  }
  return (all_deseasonal_data)
}

########################################
'%&%' <- function(x, y)paste0(x,y)

########################################
repeat_pattern <- function(time_series, max_len=108){
  old_series = time_series
  ext_series = old_series
  max_length = max_len
  
  #Extend time series
  length(ext_series) = max_length
  
  #Extend time series
  ext_series[(max_length-(length(old_series)-1)):max_length] <- old_series
  
  ##Chunk repetition
  mid_end    = max_length-(length(old_series)-1)-1
  mid_start  = (length(old_series))+1
  times      = ceiling(length(ext_series[mid_start:mid_end])/length(old_series))
  new_series = rep.int(old_series, times)
  chunk = length(ext_series[mid_start:mid_end])
  series_chunk = new_series[1:chunk]
  ext_series[mid_start:mid_end] = series_chunk
  return (ext_series)
}

###########################################
decomposition_ts <- function(cluster, freq){
  tsData_ = cluster
  tsData = ts((tsData_),  frequency = freq)
  tsData = log(tsData)
  #tsData <- replace(tsData, is.na(tsData), 0)
  
  column_names = colnames(tsData)
  decomp_data = NULL
  
  for(i in 1:length(column_names)){
    tseries = tsData[,column_names[i]]

    max_length = 108
    if (length(na.omit(tseries)) < max_length){
      #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
      #print("FILLING THE NA's")
      #extrapolating the pattern
      #tseries = ts(repeat_pattern(na.omit(tseries), max_length), frequency = freq)
      
      stl = stl(tseries, s.window = "periodic",na.action = na.remove)$time.series
      
      season    = stl[,"seasonal"]
      trend     = stl[,"trend"]
      remainder = stl[,"remainder"]
      
      season    = as.vector(t(season))
      trend     = as.vector(t(trend))
      remainder = as.vector(t(remainder))
      
      season    = c(season)
      trend     = c(trend)
      remainder = c(remainder)
  
      length(season)    <- max_length
      length(trend)     <- max_length
      length(remainder) <- max_length
  
      streched_vector = cbind(season, trend, remainder)
    }else{
      stl = stl(tseries, s.window = "periodic",na.action = na.remove)$time.series
      streched_vector = stl
    }
    
    streched_vector = as.vector(t(streched_vector))
    decomp_data = cbind(decomp_data, streched_vector)
    colnames(decomp_data)[i] = column_names[i]
  }
  return (decomp_data)
}
get_time_series <- function(clusters_){
  data_container= NULL
  for (num in 1:length(clusters_)){
    cluster = data.frame(clusters_[num])
    col = "X" %&% num %&% ".name"
    ts  = cluster[,col]
    ts = toString(ts)
    numbers = as.numeric(str_extract_all(ts, "[0-9]+")[[1]])
    data = data_all[paste0("ts",c(numbers)), ]
    drops <- c("V2")
    data = data[ , !(names(data) %in% drops)]
    data_container[[num]] <- data
  }
  return (data_container)
}
get_time_series_cclust <- function(clusters_){
  data_container= NULL
  for (num in 1:length(clusters_)){
    cluster = data.frame(clusters_[num])
    #col = "X" %&% num %&% ".name"
    #ts  = cluster[,col]
    ts = cluster$name
    ts = toString(ts)
    numbers = as.numeric(str_extract_all(ts, "[0-9]+")[[1]])
    data = data_all[paste0("ts",c(numbers)), ]
    drops <- c("V2")
    data = data[ , !(names(data) %in% drops)]
    data_container[[num]] <- data
  }
  return (data_container)
}

log_stl <- function(data_clust,freq){
  log_container = NULL
  for (num in 1:length(data_clust)){
    cluster = data.frame(data_clust[num])
    cluster = decomposition_ts(t(cluster), freq)
    log_container[[num]] <- cluster
  }
  return(log_container)
}

ts_features_preprocessing <- function(data, scaling=FALSE){
  rownames(data) <- data[,1]
  data <- data[,-1]
  
  if(scaling){
    data <- t(scale(t(data)))
    #data <- scale(data)
  }
  
  data <- data[, colSums(is.na(data)) != nrow(data)]
  data <- replace(data, is.na(data), 0)
  return (data)
}

ts_feature_calculator <- function(ts_single, freq){
  hwl <- bind_cols(
    tsfeatures(ts_single,
               c("acf_features","entropy","lumpiness",
                 "flat_spots","crossing_points")),
    tsfeatures(ts_single,"stl_features", s.window='periodic', robust=TRUE),
    tsfeatures(ts_single, "max_kl_shift", width=freq), #width changed to "freq" from 48
    tsfeatures(ts_single,
               c("mean","var"), scale=FALSE, na.rm=TRUE),
    tsfeatures(ts_single,
               c("max_level_shift","max_var_shift"), trim=TRUE)) %>%
    select(mean, var, x_acf1, trend, linearity, curvature, peak, trough,
           entropy, lumpiness, spike, max_level_shift, max_var_shift, flat_spots,
           crossing_points, max_kl_shift, time_kl_shift)
  return(hwl)
  
}
dimension_reduction <- function(data, pca=FALSE, tsne=FALSE){
  
  if (tsne){
    ## Rtsne function may take some minutes to complete...
    data = Rtsne(as.matrix(data), perplexity=4, pca=TRUE)
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
  val  <- data[-train_ind,]
  
  return (list(train=train, validation=val))
}

optimal_clusters_num <- function(data){
  nb <- NbClust(data, distance = "euclidean", min.nc = 2,max.nc = floor(dim(data)[1]/2), 
                method = "kmeans", index = 'hartigan')
  nbclusters = nb$Best.nc[1]
  print("**************************************")
  cat("Optimal Number of Clusters", nbclusters)
  return (nbclusters)
}

visualize_clusters <- function(train, clusters,kmean_pred=TRUE, kmean_ = FALSE, apclust_ = FALSE, mclust_ = FALSE){
  cat("Dimension of Training Data",dim(train))
  print("")
  #Clustering
  X <- 0
  if(kmean_pred){
    cl <- cclust(data.matrix(train), clusters, iter.max=200, dist = "euclidean", method = "kmeans")
    saveRDS(cl, file = "cluster.bin", ascii = FALSE, version = NULL,
            compress = TRUE, refhook = NULL)
    qs <- cbind(name = row.names(train),cluster = cl$cluster)
    cluster_container <- NULL
    qs <- data.frame(qs)
    for (y in 1:clusters){
      cluster_container[[y]] <- list(qs[qs$cluster == y,])
    }
    X <- cluster_container
  }
  
  if(kmean_){
    km.res <- kmeans(train, clusters, nstart = 25)
    print(km.res)
    # Visualize
    clus = fviz_cluster(km.res, data = train,
                        ellipse.type = "convex",
                        palette = "jco",
                        ggtheme = theme_minimal())
    plot(clus)
    
    #Interprating the clusters
    
    clusters = clus[1]$data
    # for (r in 1:length(clus)){
    #   print(clus[r])
    # }
    write.csv(km.res$centers,file = "Clusters.csv",row.names = FALSE,col.names = TRUE)
    X <- split(clusters, clusters$cluster)
  }
  
  if(apclust_){
    apres <- apcluster(negDistMat(r=2), train, details=TRUE)
    
    ## show details of clustering results
    show(apres)
    
    ## plot clustering result
    # plot(apres,train)
    
    X <- apres
  }
  if(mclust_){
    BIC <- mclustBIC(train)
    mod <- Mclust(train, x = BIC)
    #clus <- fviz_mclust(mod7, "classification", geom = "point", pointsize = 1.5, palette = "jco")
    clus = fviz_cluster(mod, data = train,
                        ellipse.type = "convex",
                        palette = "jco",
                        ggtheme = theme_minimal())
    plot(clus)
  }
  return (X)
}


#measure extractor from time series
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

#feature extractor from time series
ts_feature_extrator <- function(similar_ts, ts_freq){
  #Extracting feature vector using the tsmeasures() in tsfeature package.
  feature_mat = NULL
  for(k in 1 :ncol(similar_ts)){
    input_series = similar_ts[,k]
    input_series = input_series[!is.na(input_series)]
    input        = ts(input_series ,freq=ts_freq)
    y            = ts_feature_calculator(input, ts_freq)
    feature_mat  = rbind(feature_mat,y)
  }
  rownames(feature_mat) = colnames(similar_ts)
  write.csv(data.frame(feature_mat),file = "extracted_ts_feature.csv",row.names = TRUE,col.names = TRUE)
  features = read.csv("extracted_ts_feature.csv", header = TRUE, sep=',')
  return(features)
}

ts_distinct_data <- function(data_all,freq){
  unique_freq <- freq
  freq_col    <- data_all$V2
  similar_ts  <- data.frame(t(data_all))[freq_col == unique_freq]
  similar_ts  <- similar_ts[-1, ] 
  #write.csv(similar_ts,file = paste(toString(freq),"_ts.csv", sep=""))
  return(similar_ts)
}



data_preprocessing <- function(csv_file, train = TRUE){
  csv_file_ = csv_file
  csv_file <- t(csv_file)
  if (train){
    csv_file <- csv_file[-c(3), ]
  }
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

cluter_prediction <- function(ts_input, freq_){
  
  ts_ <- ts_input
  ts_ <- ts(ts_,freq = freq_)
  f_ts_ <- t(scale(t(data.frame(ts_feature_calculator(ts_)))))
  cl1 <- readRDS("cluster.bin", refhook = NULL)
  ycl<-predict(cl1, data.matrix(f_ts_))
  
  return (ycl$cluster)
}

find_cluster_no <- function(cluster,time_series_name){
  for (p in 1:length(clusters_)){
    if (time_series_name %in% data.frame(cluster[[p]])$name){
      return(p)
    }
  }
}

Prediction_ts <- function(ts_input,ts_output,freq,clust_no,model){
  input_container <- NULL
  history <- NULL
  forcast_ts <-NULL
  pred_out <- NULL
  
  output_window = freq
  input_window  = floor(output_window*(1.25))
  
  ####### predicing cluster ########
  #cluster = cluter_prediction(ts_input, freq)
  
  cluster = clust_no
  
  ######################## Pre-Processing ######################
  #1 Assigning Cluster to Input Series
  input_container[[1]] <- data.frame(t(ts_input))
  
  #2 Decomposition of Input Series
  decomposed_data <- log_stl(input_container,freq)
  
  #3 Deseasonlization of Input Series
  deseasonal <- remove_season(decomposed_data, freq)
  
  #Seasonality factor
  season = deseasonalization(as.vector(unlist(decomposed_data)),
                             c('seasonal', 'trend', 'remainder'),
                             freq,prediction=TRUE)
  #4 sample of x_train [1, 15]
  length_history   = dim(ts_input)[1] #length of train ts
  input_pointer    = length_history-input_window  + 1 #input pointer
  output_pointer   = length_history-output_window + 1 #output pointer
  
  #deseasonalized singal input
  des_data <- data.frame((t(deseasonal[[1]]))[,input_pointer:length_history])
  
  #normalized vector and input
  normalize_ts <- normalization(t(des_data))
  test_x <- k$eval(k_expand_dims((normalize_ts$X),axis=3))
  
  ################# Model Prediction ##################
  # file_name <- paste(cluster, ".h5", sep="")
  # model <- load_model_hdf5(file_name, custom_objects = NULL, compile = TRUE)
  pred_out <- model %>%
    predict(test_x, batch_size = 1)
  
  ################# Post Processing ##################
  pred_out = as.vector(unlist(pred_out, use.names = FALSE)) + 
    as.vector(unlist(normalize_ts$norm_vector, use.names = FALSE))
  
  
  pred_out = as.vector(unlist(pred_out)) + as.vector(unlist(season)[output_pointer:length_history])
  
  #print(exp(as.vector(unlist(season)[output_pointer:length_history])))
  plot(c(unlist(ts_output)),type="b", col="green",xlab = "Months",ylab = "Forecasted Amount",main = "ATM/Branch Forecasting")
  lines(c(unlist(exp(pred_out))),col ="red",type = 'b')
  return(pred_out)
}

######################### Running Prediction #####################################
prediction_accuracy <- function(similar_ts, freq, test_file = "cif-results.txt"){
  csv_test_file_ <- read.csv(test_file, sep=";", header = FALSE)
  data_test = data_preprocessing(csv_test_file_,train = FALSE)
  data_test <- data.frame(t(data_test))
  errors <- NULL
  ts_input <- NULL
  ts_forecast <- NULL
  model <- NA
  file_name <- NULL
  
  output_window = freq
  input_window  = floor(output_window*1.25)
  
  #iteration for each time series
  count = 0
  for (name in colnames(similar_ts)){
    #print(name)
    ts_name <- name
    input_series = similar_ts[,ts_name]
    input_series = input_series[!is.na(input_series)]

    data_for_test <- data_test[,ts_name]
    data_for_test <- data_for_test[!is.na(data_for_test)]
    input_data <- data.frame(c(input_series,data_for_test))
    colnames(input_data) <- ts_name
    clust = find_cluster_no(clusters_,ts_name)
    model_name <- paste("model",clust, sep="")
    if (is.na(model[clust])){
      model[clust] = 1
      file_name <- paste(clust, freq, ".h5", sep="")
      assign(model_name,load_model_hdf5(file_name, custom_objects = NULL, compile = TRUE))
      #print("####*******#########******########******########")
    }
    history_data = data.frame(c(input_series))
    test_data    = data.frame(c(data_for_test))
    predicted_forecast = data.frame(Prediction_ts(history_data,test_data,freq,clust_no = clust,model = eval(parse(text=model_name))))
    lo = dim(input_data)[1]
    l1 = dim(input_data)[1] - output_window + 1
    l2 = l1 - input_window + 1

    error = errorMetric(c(unlist(exp(predicted_forecast))),c((input_data)[l1:lo,])
                        , type = "sAPE", statistic = "M")
    ts_input <- data.frame(c(unlist(ts_input,use.names = FALSE),unlist((input_data)[l1:lo,],use.names = FALSE)))
    ts_forecast <- data.frame(c(unlist(ts_forecast,use.names = FALSE),unlist(exp(predicted_forecast),use.names = FALSE)))
    
    
    forecast_comparison = cbind(actual = ts_input, forecasted = ts_forecast)
    
    colnames(forecast_comparison)[1] = "Actual"
    colnames(forecast_comparison)[2] = "Forecast"
    #View(forecast_comparison)
    errors[name] <- error
    cat(ts_name, "************", error)
    #print("")
    count = count + 1
  }
  
  
  #SMAPE Error calculation
  print("**************** SMAPE SCORE ******************")
  print(sum(errors)/length(errors))
  print(errorMetric(c(unlist(ts_forecast)),c(unlist(ts_input)), type = "sAPE", statistic = "M"))
  
  return (sum(errors)/length(errors))
}

prediction_accuracy_of_cluster <- function(test_file = "cif-results.txt", cluster_no = 0){
  csv_test_file_ <- read.csv(test_file, sep=";", header = FALSE)
  data_test = data_preprocessing(csv_test_file_,train = FALSE)
  data_test <- data.frame(t(data_test))
  errors <- NULL
  ts_input <- NULL
  ts_forecast <- NULL
  
  #iteration for each time series
  count = 0
  for (name in colnames(similar_ts)){
    ts_name <- name
    
    input_series = similar_ts[,ts_name]
    input_series = input_series[!is.na(input_series)]

    input_data <- data.frame(c(input_series,data_test[,ts_name]))
    colnames(input_data) <- ts_name
    clust = find_cluster_no(clusters_,ts_name)
    if(cluster_no>0 & clust==cluster_no){
      #print(length(input_series))
      history_data = data.frame(c(input_series))
      test_data    = data.frame(c(data_test[,ts_name]))
      predicted_forecast = data.frame(Prediction_ts(history_data,test_data,12,clust_no = clust))
      lo = dim(input_data)[1]
      l1 = dim(input_data)[1] - 12 + 1
      l2 = l1 - 15 + 1
      error = errorMetric(c(unlist(exp(predicted_forecast))),c((input_data)[l1:lo,])
                          , type = "sAPE", statistic = "M")
      ts_input <- data.frame(c(unlist(ts_input,use.names = FALSE),unlist((input_data)[l1:lo,],use.names = FALSE)))
      ts_forecast <- data.frame(c(unlist(ts_forecast,use.names = FALSE),unlist(exp(predicted_forecast),use.names = FALSE)))
      
      
      forecast_comparison = cbind(actual = ts_input, forecasted = ts_forecast)
      
      colnames(forecast_comparison)[1] = "Actual"
      colnames(forecast_comparison)[2] = "Forecast"
      #View(forecast_comparison)
      errors[name] <- error
      cat(ts_name, "************", errors[name])
      print("")
      count = count + 1
    }
  }
  
  
  #SMAPE Error calculation
  print(sum(errors)/length(errors))
  print(errorMetric(c(unlist(ts_forecast)),c(unlist(ts_input)), type = "sAPE", statistic = "M"))
}

############################### MAIN LOOP ######################################

#Reading Dataset of TS
csv_file_ <- read.csv("cif-dataset.txt", sep=";", header = FALSE)
data_all = data_preprocessing(csv_file_)
total_freq = distinct(select(data_all, 'V2'))

#spliting dataframe on basis of frequencies
for(i in 1:(ncol(total_freq)+1)){
  similar_ts <- ts_distinct_data(data_all,total_freq[i,1])
  extracted_ts_feature <- ts_feature_extrator(similar_ts, total_freq[i,1])
  data  <- ts_features_preprocessing(extracted_ts_feature, scaling=TRUE)
  data  = dimension_reduction(data, pca=TRUE, tsne=FALSE)
  data  = train_test_split(data)
  train = data$train
  test  = data$train
  clusters_ = visualize_clusters(train, optimal_clusters_num(train), kmean_pred = FALSE, kmean_ = TRUE)
  #clusters_ = visualize_clusters(train, 1, kmean_pred = FALSE, kmean_ = TRUE)
  
  data  = get_time_series(clusters_)
  #data = get_time_series_cclust(clusters_)
  detrended_data <- log_stl(data,total_freq[i,1])
  deseasonal = remove_season(detrended_data, total_freq[i,1])
  frame_normalized_data <- framing_dataset(deseasonal, total_freq[i,1])

  batch_size = 4
  cluster_train = 0
  for (j in 1:length(frame_normalized_data)){
    if(j>cluster_train){
      x_train <- data.frame(frame_normalized_data[[j]]$norm_X)
      y_train <- data.frame(frame_normalized_data[[j]]$norm_Y)
      batch_train = train_validation_sets(x_train, y_train, total_freq[i,1])
      val_data    = batch_data(batch_train$val_x, batch_train$val_y, batch_size, total_freq[i,1])
      train_data  = batch_data(batch_train$train_x, batch_train$train_y, batch_size, total_freq[i,1])
      LSTM(train_data$train_x, train_data$train_y, val_data$train_x, val_data$train_y, batch_size,j, total_freq[i,1])
    }
  }
  prediction_accuracy(similar_ts, total_freq[i,1])
  break
}
#prediction_accuracy_of_cluster(cluster_no = cluster_train)