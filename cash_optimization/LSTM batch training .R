library("keras") 
library("forecTheta")

k <- backend()
use_condaenv("r-tensorflow")
train_x = "x_train.csv"
train_y = "y_train.csv"

train_x  <- k$eval(k_expand_dims(data.matrix(read.csv(train_x, sep=",", header = TRUE)), axis = 3))
#train_y  <- k$eval(k_expand_dims(data.matrix(read.csv(train_y, sep=",", header = TRUE)), axis = 3))
train_y <- data.matrix(read.csv(train_y, sep=",", header = TRUE))

cat("Training Data: ", dim(train_x), "\n")
cat("Testing Data: ",  dim(train_y), "\n")

input_shape <- c(dim(train_x)[1], dim(train_x)[2],dim(train_x)[3])
n           <- input_shape[1]
t_steps     <- input_shape[2]
features    <- input_shape[3]
batch_size  <- 41
epochs      <- 50
cat(n, t_steps, features, "\n")

cat('Creating model:\n')
model <- keras_model_sequential()

model %>%
  layer_lstm(units = 80, batch_input_shape = c(batch_size,t_steps, features), stateful = TRUE) %>%
  layer_dense(units = dim(train_y)[2], use_bias=FALSE, bias_initializer='zeros')

model %>% compile(loss = 'mae', optimizer = 'adam')
summary(model)

cat('Training\n')
par(mfrow=c(2,1)) 

smape_score <- list()
for (i in 1:epochs) {
  model %>% fit(x = train_x,
                y = train_y,
                batch_size = batch_size,
                epochs = epochs,
                verbose = 1,
                shuffle = TRUE)
  model %>% reset_states()
  
  cat('Testing\n')
  test_x = train_x[1:batch_size,,]
  test_x <- k$eval(k_expand_dims(test_x,axis=3))
  print(dim(test_x))
  
  #Model Prediction
  pred_out <- model %>%
    predict(test_x, batch_size = batch_size)
  
  #Calculation of Error
  error = errorMetric(c(train_y[1:batch_size,]), c(pred_out), type = "sAPE", statistic = "M")
  smape_score[i] = error
  
  #Plot Graphs
  plot(c(pred_out),type="l", col="red", pch = 1, xlab="Days", ylab="Amount")
  lines(c(train_y[1:batch_size,]),col="green")
  error = c(unlist(smape_score, use.names=FALSE))
  plot(error, type="b", col="black", pch = 2, xlab="Epochs", ylab="SMAPE Error")
  cat("\n *********************************** \n Epoch: ", i, "\n \n")
}