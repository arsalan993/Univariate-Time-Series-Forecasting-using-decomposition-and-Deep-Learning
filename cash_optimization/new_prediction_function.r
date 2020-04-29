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


Prediction_ts <- function(ts_input, pred_end, freq,clust_no){
  input_container <- NULL
  history_norm <- NULL
  history_denorm <- NULL
  pred_out <- NULL
  y_label <- NULL
  
  ####### predicing cluster ########
  cluster = cluter_prediction(ts_input, freq)
  #cluster = clust_no
  ##################################
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
  lo = dim(ts_input)[1]
  l1 = dim(ts_input)[1] -12 + 1
  l2 = l1 - 15 
  j = 12
  
  history_denorm <- data.frame(c(NULL,as.vector(ts_input[1:15,])))
  des_data <- data.frame(t(deseasonal[[1]]))[,1:(15)]
  normal_in <- normalization(des_data)
  history_norm   <- data.frame(c(NULL,as.vector((normal_in$X)[1:15])))
  
  
  for (i in 1:l2){
    #des_data <- data.frame((t(deseasonal[[1]]))[,i:(i+14)])

    #normalize_ts <- normalization(t(des_data))

    #denormalize  <- normalization(normalize_ts$X,norm = FALSE, normalizer = normalize_ts$norm_vector)

    #test_x <- k$eval(k_expand_dims((normalize_ts$X),axis=3))

    if (i > 1){
      des_data <- data.frame(log(history_denorm))[,i:(i+14)]
      normal_in <- normalization(t(des_data))
    }

    in_norm_data <- data.frame(history_norm[,i:(i+14)])
    test_x <- k$eval(k_expand_dims(in_norm_data,axis=3))
    file_name <- paste(cluster, ".h5", sep="")
    model <- load_model_hdf5(file_name, custom_objects = NULL, compile = TRUE)
    pred_out <- model %>%
      predict(test_x, batch_size = 1)

    # ############### Post Processing ##################
    #y_label <- data.frame((t(deseasonal[[1]]))[,l1:lo])
    if (i != l2){
      history_norm <- data.frame(c(as.vector(history_norm),pred_out[1]))
    }
    else{
      history_norm <- data.frame(c(as.vector(history_norm),as.vector(pred_out)))
    }
    pred_out = as.vector(unlist(pred_out, use.names = FALSE)) +
      as.vector(unlist(normal_in$norm_vector, use.names = FALSE))

    #y_label = as.vector(unlist(y_label)) + as.vector(unlist(season)[l1:lo])
    i1 = i  + 15
    i2 = i1 + 12
    pred_out = as.vector(unlist(pred_out)) + as.vector(unlist(season)[i1:i2])
    if (i != l2){
      history_denorm <- data.frame(c(unlist(as.vector(history_denorm)),exp(pred_out[1])))
    }else{
      history_denorm <- data.frame(c(unlist(as.vector(history_denorm)),exp(pred_out)))
    }
    pred_out <- NULL
    # lo = lo - 1
    # l1 = l1 - 1
    # l2 = l2 - 1
    # j =  j -  1
  }
  #   
  # 
  # # plot(c(unlist(y_label)),type="l", col="green")
  # # lines(c(unlist(pred_out)),col ="red")
  # plot(c(unlist(exp(ts_input))),type="b", col="green",xlab = "Months",ylab = "Forecasted Amount",main = "ATM/Branch Forecasting")
  # # lines(c(unlist(exp(history))),col ="red",type = 'b')
  # #print(cbind(exp(pred_out), exp(y_label), ts_input[16:27,]))
  # return(history)
}
csv_test_file_ <- read.csv("cif-results.txt", sep=";", header = FALSE)
data_test = data_preprocessing(csv_test_file_,train = FALSE)
data_test <- data.frame(t(data_test))
#test_freq = distinct(select(data_test, 'V2'))
input_data <- data.frame(ts11 = c(similar_ts$ts11,data_test$ts11))
clust = 1
predicted_forecast = data.frame(Prediction_ts(input_data,12,clust_no = clust))
# lo = dim(input_data)[1]
# l1 = dim(input_data)[1] -12 + 1
# l2 = l1 - 15 +1
# error = errorMetric(c(unlist(exp(predicted_forecast))),c((input_data)[l1:lo,])
#                     , type = "sAPE", statistic = "M")
# print(error)
# compare = cbind(c(unlist(exp(predicted_forecast))),c((input_data)[l1:lo,]))
# View(compare)