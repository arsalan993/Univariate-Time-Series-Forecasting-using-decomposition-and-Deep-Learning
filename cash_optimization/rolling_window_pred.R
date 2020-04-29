library("keras") 
library(zoo)
library(lubridate)
library(xts)
library(ggfortify)
use_condaenv("r-tensorflow")
k <- backend()
#https://stackoverflow.com/questions/29641693/how-to-find-scaling-factor-for-new-data-in-r

cluter_prediction <- function(ts_input, freq_){
  ts_ <- ts_input
  ts_ <- ts(ts_,freq = freq_)
  f_ts_ <- t(scale(t(data.frame(ts_feature_calculator(ts_)))))
  print(f_ts_)
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

Prediction_ts <- function(ts_input, pred_end, lag, freq, cluster){
  input_container <- NULL

  file_name <- paste(cluster, ".h5", sep="")
  model <- load_model_hdf5(file_name, custom_objects = NULL, compile = TRUE)
  print("*****************************")
  cat("Model Loaded: ", file_name, "\n")
  
  ####### predicing cluster ########
  #cluster = cluter_prediction(ts_input, freq)

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
  out_window = 12
  in_window = out_window*1.25
  input_length = dim(ts_input)[1]
  
  recent_timestamp  = (input_length-in_window)+1
  print(recent_timestamp)
  print(input_length)
  unlist_deseasonal = unlist(deseasonal, use.names = FALSE)
  raw_input = data.frame(unlist_deseasonal[recent_timestamp:input_length])
  #raw_input = data.frame(unlist_deseasonal[1:15])
  #forecasted_ts = data.frame(c(unlist(raw_input, use.names = FALSE)))
  
  #Not adding input to forecast history
  forecasted_ts <- NULL
  
  
  #Creating seasonality of length 12 from last 12 values of input
  #unlist_season <- unlist(season, use.names = FALSE)
  added_season <- season[(input_length - out_window + 1):input_length]
  print(added_season)
  
  loop_end = pred_end/lag
  for (i in 1:loop_end) {
    
    #Pre-Processing
    norm_data <- normalization(t(raw_input))
    normalized_input <- norm_data$X
    normalizer_value <- norm_data$norm_vector
    
    #Prediction
    model_input <- k$eval(k_expand_dims(normalized_input,axis=3))
    model_output <- model %>%
      predict(model_input, batch_size = 1)
    
    #Post-processing
    denormalized_output = model_output+normalizer_value

    raw_input = data.frame(c(unlist(raw_input[(lag+1):in_window,], use.names = FALSE), 
                 unlist(denormalized_output[,1:lag], use.names = FALSE)))
    
    logged_out <- as.vector(unlist(denormalized_output)) + as.vector(unlist(added_season))
    print(logged_out)
    forecasted_ts = data.frame(c(unlist(forecasted_ts, use.names = FALSE), 
                                 unlist(logged_out[1:lag], use.names = FALSE)))
    print(as.vector(raw_input))
    cat("count = ",i,"\n")
  }
  
  # plot(c(unlist(forecasted_ts)),type="l", col="green")
  # print(dim(forecasted_ts))
  
  value = unlist(forecasted_ts, use.names = FALSE)
  #value =  value[1:(pred_end+out_window)]
  #value <- as.vector(unlist(value)) + as.vector(unlist(season))
  return (value)
}



season_extractor <- function(inp_timeseries){
 return  (deseasonalization(as.vector(unlist(inp_timeseries)),
                    c('seasonal', 'trend', 'remainder'),
                    freq,prediction=TRUE))
}

#########################################################################################
ts_name <- 'ts63'
frequency_ <- 12
csv_test_file_ <- read.csv("cif-results.txt", sep=";", header = FALSE)
data_test = data_preprocessing(csv_test_file_,train = FALSE)
data_test <- data.frame(t(data_test))
#test_freq = distinct(select(data_test, 'V2'))

input_series = similar_ts[,ts_name]
input_series = input_series[!is.na(input_series)]
cat(ts_name,length(input_series))
print("")

input_data <- data.frame(c(input_series,data_test[,ts_name]))
input_data_ <- ts(input_data,frequency = 12,start = c(2015,1))
input_data_ts_index <- as.Date.yearmon(time(input_data_))

clust = find_cluster_no(clusters_,ts_name)
lag <- 12
chunk_start  = 12
days_to_forecast <- 12 ## multiple of 12

chunk <- data.frame(input_data[1:length(input_series),])
row.names(chunk) <- input_data_ts_index[1:length(input_series)]
colnames(chunk) <- "Input Time Series"

old_last_date <- as.Date(row.names(chunk)[length(input_series)])
month(old_last_date) <- month(old_last_date) + 2

first_chunk_date <- as.Date(row.names(chunk)[1])

value = data.frame(Prediction_ts(chunk, days_to_forecast, lag, frequency_, clust))
Forecasted_Time_Series <- ts(exp(value),frequency = frequency_, start=c(year(old_last_date),
                                                     month(old_last_date)))

input_Time_Series <- ts(chunk,frequency = frequency_, start = c(year(first_chunk_date),
                                                        month(first_chunk_date)))

True_Label_Time_Series <- ts(data_test[,ts_name],frequency = frequency_, start=c(year(old_last_date),
                                                                            month(old_last_date)))

ts_matrix <- ts.union(True_Label_Time_Series,Forecasted_Time_Series)

True_label_frame <- data.frame(True_Label_Time_Series,
                               row.names = as.Date.yearmon(time(True_Label_Time_Series)))
colnames(True_label_frame) <- "True Labels"

predicted_frame <- data.frame(Forecasted_Time_Series,
                               row.names = as.Date.yearmon(time(Forecasted_Time_Series)))
colnames(predicted_frame) <- "Forecasted Time Series"

# View(chunk)
# View(True_label_frame)
# View(predicted_frame)

pl <- autoplot(as.xts(ts_matrix),facets = FALSE,ts.scale = TRUE)
plot(pl)