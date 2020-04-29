library("keras") 
library("forecTheta")

k <- backend()
use_condaenv("r-tensorflow")

train_x = "x_train.csv"
train_y = "y_train.csv"

train_x <- k$eval(k_expand_dims(data.matrix(read.csv(train_x, sep=",", header = TRUE)), axis = 3))
train_y <- data.matrix(read.csv(train_y, sep=",", header = TRUE))

cat("Training Data: ", dim(train_x), "\n")
cat("Testing Data: ",  dim(train_y), "\n")

input_shape <- c(dim(train_x)[1], dim(train_x)[2],dim(train_x)[3])
n           <- input_shape[1]
t_steps     <- input_shape[2]
features    <- input_shape[3]
batch_size  <- 1
epochs      <- 5
cat(n, t_steps, features, "\n")

model <- keras_model_sequential()

model %>%
  layer_lstm(units            = 50, 
             input_shape      = c(t_steps, features), 
             batch_size       = batch_size,
             return_sequences = TRUE, 
             stateful         = TRUE) %>% 
  
  layer_lstm(units            = 50,
             return_sequences = FALSE,
             stateful         = TRUE) %>%
  
  layer_dense(units = dim(train_y)[2])

model %>% 
  #compile(loss = 'mae', optimizer = 'adam', metrics = c('accuracy'))
  compile(loss = 'mae', optimizer = 'adam')
summary(model)

cat('Training\n')
par(mfrow=c(2,1)) 
smape_score <- list()

for (i in 1:epochs) {
  model %>% fit(x          = train_x, 
                y          = train_y, 
                batch_size = batch_size,
                epochs     = epochs, 
                verbose    = 1, 
                shuffle    = TRUE,
                validation_split = .20)
  
  model %>% reset_states()
  
  cat('Testing\n')
  test_x = train_x[1,,]
  test_x <- k$eval(k_expand_dims(test_x,axis=1))
  test_x <- k$eval(k_expand_dims(test_x,axis=3))
  print(dim(test_x))
  
  #Model Prediction
  pred_out <- model %>%
    predict(test_x, batch_size = batch_size)
  
  #Calculation of Error
  error = errorMetric(c(train_y[1,]), c(pred_out), type = "sAPE", statistic = "M")
  smape_score[i] = error
  
  #Plot Graphs
  plot(c(pred_out),type="l", col="red", pch = 1, xlab="Days", ylab="Amount")
  lines(c(train_y[1,]),col="green")
  error = c(unlist(smape_score, use.names=FALSE))
  plot(error, type="b", col="black", pch = 2, xlab="Epochs", ylab="SMAPE Error")
  cat("\n *********************************** \n Epoch: ", i, "\n \n")
}