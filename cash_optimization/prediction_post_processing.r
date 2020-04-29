library("keras") 
use_condaenv("r-tensorflow")
k <- backend()
#https://stackoverflow.com/questions/29641693/how-to-find-scaling-factor-for-new-data-in-r

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

Prediction_ts <- function(ts_input,freq,clust_no){
  input_container <- NULL
  history <- NULL
  forcast_ts <-NULL
  pred_out <- NULL
  
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
  lo = dim(ts_input)[1] #length of test+train ts
  l1 = dim(ts_input)[1] -12 + 1 #training ts length
  l2 = l1 - 15 +1 #model input
  
  #deseasonalized singal input
  des_data <- data.frame((t(deseasonal[[1]]))[,l2:l1])
  
  #normalized vector and input
  normalize_ts <- normalization(t(des_data))
  
  test_x <- k$eval(k_expand_dims((normalize_ts$X),axis=3))
  
  ################# Model Prediction ##################
  file_name <- paste(cluster, ".h5", sep="")
  model <- load_model_hdf5(file_name, custom_objects = NULL, compile = TRUE)
  pred_out <- model %>%
    predict(test_x, batch_size = 1)

  ################# Post Processing ##################
  y_label <- data.frame((t(deseasonal[[1]]))[,l1:lo])
  
  pred_out = as.vector(unlist(pred_out, use.names = FALSE)) + 
    as.vector(unlist(normalize_ts$norm_vector, use.names = FALSE))

  y_label = as.vector(unlist(y_label)) + as.vector(unlist(season)[l1:lo])
  pred_out = as.vector(unlist(pred_out)) + as.vector(unlist(season)[(l1-12):(lo-12)])

  plot(c(unlist(exp(y_label))),type="b", col="green",xlab = "Months",ylab = "Forecasted Amount",main = "ATM/Branch Forecasting")
  lines(c(unlist(exp(pred_out))),col ="red",type = 'b')
  return(pred_out)
}

######################### Running Prediction #####################################
prediction_accuracy <- function(test_file = "cif-results.txt"){
  
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
    input_data <- data.frame(c(similar_ts[,ts_name],data_test[,ts_name]))
    colnames(input_data) <- ts_name
    clust = find_cluster_no(clusters_,ts_name)
    predicted_forecast = data.frame(Prediction_ts(input_data,12,clust_no = clust))
    lo = dim(input_data)[1]
    l1 = dim(input_data)[1] - 12 + 1
    l2 = l1 - 15 + 1
    error = errorMetric(c(unlist(exp(predicted_forecast))),c((input_data)[l1:lo,])
                        , type = "sAPE", statistic = "M")
    ts_input <- data.frame(c(unlist(ts_input,use.names = FALSE),unlist((input_data)[l1:lo,],use.names = FALSE)))
    ts_forecast <- data.frame(c(unlist(ts_forecast,use.names = FALSE),unlist(exp(predicted_forecast),use.names = FALSE)))
    
    errors[name] <- error
    count = count + 1
  }
  
  #SMAPE Error calculation
  print(sum(errors)/length(errors))
  print(errorMetric(c(unlist(ts_forecast)),c(unlist(ts_input)), type = "sAPE", statistic = "M"))
}