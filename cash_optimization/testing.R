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
    #return(x[,"seasonal"])
    return (x[,"seasonal"]+x[,"trend"] + x[,"remainder"])
  }else{
    return(x[,"trend"] + x[,"remainder"])
  }
}

chunk <- function(x, columns) {
  n = length(columns)
  
  if((length(x)%%n)==0) {
    x = t(matrix(x, nrow=n))
  } 
  else {
    x = t(matrix(append(x, rep(NA, n-(length(x)%%n))), nrow=n))
  } 
  colnames(x) = columns
  return(x)
}


decomposition_ts <- function(cluster, freq){
  drops <- c("X")
  tsData_ = tsData_[ , !(names(tsData_) %in% drops)]
  
  tsData = ts(tsData_,  frequency = freq)
  tsData = log(tsData)
  tsData <- replace(tsData, is.na(tsData), 0)
  column_names = colnames(tsData)
  decomp_data = NULL
  
  for(i in 1:length(column_names)){
    tseries = tsData[,column_names[i]]
    print(paste0(i, ": Length of time series =", length(tseries)))
    stl = stl(tseries, s.window = "periodic", na.action=na.contiguous)$time.series
    print(stl)
    decomp_data = cbind(decomp_data, as.vector(t(stl)))
    data = data.frame(chunk(decomp_data, colnames(stl)))
    colnames(decomp_data)[i] = column_names[i]
    stl_plot = cbind(stl, tseries)
    
    print(class(stl))
    plot(stl_plot)
    break
  }
  #colnames(decomp_data) = column_names
  return (decomp_data)
}

file = "cluster12.csv"
tsData_ <- read.csv(file, sep=",", header = TRUE)
#tsData_ = c(1,2,3,5,2,3,5,5,8,9,5,8,45,1,568,4,2,5,5,2,3,5,5,8,5,8,45,1,568,4,2,5,5,5,5,8,5,8,45,1,568,4,2,5,5,58,152,156,789,1235,4554,456456,151,5,5,58,15,6,67,21,1,87,43,34)
#tsData_ = data.frame(tsData_)
tsData_[[1]] = tsData_
print(class(tsData_))
data    = decomposition_ts(tsData_, 12)
print(dim(data))

seasonality = deseasonalization(data,
                                c('seasonal', 'trend', 'remainder'),
                                freq,prediction=TRUE)
seasonality = data.frame(seasonality)
# print(exp(seasonality))