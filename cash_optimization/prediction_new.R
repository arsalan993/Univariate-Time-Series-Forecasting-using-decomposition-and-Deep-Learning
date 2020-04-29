library("keras") 
library("forecTheta")

k <- backend()
use_condaenv("r-tensorflow")
train_x = "x_train.csv"
train_y = "y_train.csv"

train_x  <- k$eval(k_expand_dims(data.matrix(read.csv(train_x, sep=",", header = TRUE)), axis = 3))
train_y <- data.matrix(read.csv(train_y, sep=",", header = TRUE))

cat("Training Data: ", dim(train_x), "\n")
cat("Testing Data: ",  dim(train_y), "\n")


index_pred = 1
cat('Testing\n')
test_x = train_x[index_pred,,]
test_x <- k$eval(k_expand_dims(test_x,axis=1))
test_x <- k$eval(k_expand_dims(test_x,axis=3))
print(dim(test_x))

new_model  = load_model_hdf5("final model.h5", compile = TRUE, custom_objects = NULL)
batch_size = 1

#Model Prediction
pred_out <- new_model %>%
  predict(test_x, batch_size = batch_size)

#Calculation of Error
error = errorMetric(c(train_y[index_pred,]), c(pred_out), type = "sAPE", statistic = "M")
print(error)

#Plot Graphs
plot(c(pred_out),type="l", col="red", pch = 1, xlab="Months", ylab="Amount")
lines(c(train_y[index_pred,]),col="green")