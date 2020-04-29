ts_ <- similar_ts$ts10
ts_ <- ts(ts_,freq = 12)
f_ts_ <- data.frame(ts_feature_calculator(ts_))
f_ts_ <- t(scale(t(f_ts_)))
cl1 <- readRDS("cluster.bin", refhook = NULL)
ycl<-predict(cl1, data.matrix(f_ts_))







# library("keras") 
# use_condaenv("r-tensorflow")
# k <- backend()
# 
# X_n = "x_train.csv"
# Y_n= "y_train.csv"
# X <- data.matrix(read.csv(X_n, sep=",", header = TRUE))
# Y <- data.matrix(read.csv(Y_n, sep=",", header = TRUE)) 
# train_x <- X[1:dim(X)[1]-1,] 
# train_y <- Y[1:dim(X)[1]-1,]
# 
# train_x <- k$eval(k_expand_dims(train_x, axis = 3))
# #train_y <- data.matrix(read.csv(train_y, sep=",", header = TRUE))
# 
# cat("Training Data: ", dim(train_x), "\n")
# cat("Testing Data: ",  dim(train_y), "\n")
# 
# input_shape <- c(dim(train_x)[1], dim(train_x)[2],dim(train_x)[3])
# n           <- input_shape[1]
# t_steps     <- input_shape[2]
# features    <- input_shape[3]
# batch_size  <- 1
# epochs      <- 50
# cat(n, t_steps, features, "\n")
# 
# model <- keras_model_sequential()
# 
# model %>%
#   layer_lstm(units            = 50, 
#              input_shape      = c(t_steps, features), 
#              batch_size       = batch_size,
#              return_sequences = TRUE, 
#              stateful         = TRUE) %>% 
#   
#   layer_lstm(units            = 50,
#              return_sequences = FALSE,
#              stateful         = TRUE) %>%
#   
#   layer_dense(units = dim(train_y)[2])
# 
# 
# 
# model %>% 
#   compile(loss = 'mae', optimizer = 'adam', metrics = c('accuracy'))
# 
# summary(model)
# 
#   
#   
#   
# for (i in 1:epochs) {
#   model %>% fit(x          = train_x, 
#                 y          = train_y, 
#                 batch_size = batch_size,
#                 epochs     = epochs, 
#                 verbose    = 1, 
#                 shuffle    = TRUE,
#                 validation_split = .20)
#   
#   model %>% reset_states()
#   cat("Epoch: ", i)
# }
# #########saving model############
# save_model_hdf5(model, "ts52_ts51.h5", overwrite = TRUE,
#                 include_optimizer = TRUE)
# 
# ##### Testing
# test_x_ <- X[dim(X)[1],] 
# test_y_ <- Y[dim(X)[1],]
# test_x <- k$eval(k_expand_dims(test_x_,axis=1))
# test_x <- k$eval(k_expand_dims(test_x,axis=3))
# pred_out <- model %>%
#   predict(test_x, batch_size = batch_size)
# 
# plot(c(pred_out),type="l", col="red")
# lines(c(test_y_),col="green")

################################################
# 
# normalization <- function(X,Y,norm = TRUE){
#   inp_win = dim(X)[2]
#   out_win = dim(Y)[2]
#   cmb <- cbind(X,Y)
#   normalizer = cmb[,inp_win]
#   if(norm){
#     norm_cmb <- cmb - normalizer
#     norm_cmb[,inp_win] <- normalizer
#   }
#   else{
#     norm_cmb <- cmb + normalizer
#     norm_cmb[,inp_win] <- normalizer
#   }
#   ind = inp_win+1
#   return(list(X=norm_cmb[,1:inp_win],Y=norm_cmb[,ind: out_win]))
# }
# 
# 
# framing_dataset <- function(deseasonal_){
# 
#   for (clust in deseasonal_){
#     clust <- data.frame(clust)
#     col_names <- colnames(clust)
#     data_framing(clust)
#     
#     # break
#   }
# 
# }
# 
# ############################################3
# data_framing <- function(dataset,input_window=15,output_window = 12){
# # c1 <- c(1,2,3,4,5,6,7,8,9,10,11,12)
# # c2 <- c1 * 10
# #c3 <- c1 * 20
#   inp_win = input_window
#   out_win = output_window
#   trend_train <- dataset
#   trend_train = data.matrix(trend_train)
#   end_index = length(trend_train[,1])-inp_win-out_win+1
#   start_y = inp_win+1
#   m1 <- matrix(1:length(trend_train),nrow = dim(trend_train)[1],
#               ncol = dim(trend_train)[2])[1:end_index,]
#   X <- t(sapply(m1, function(x) trend_train[x:(x + inp_win -1)]))
#   array_X <- array(t(X), dim = c(inp_win, end_index,dim(trend_train)[2]))
#   m2 <- matrix(1:length(trend_train),nrow = dim(trend_train)[1],
#                ncol = dim(trend_train)[2])[1:end_index,]
#   Y <- t(sapply(m2,function(x) trend_train[x:(x + out_win - 1)]))
#   array_Y <- array(t(Y), dim = c(out_win, end_index,dim(trend_train)[2]))
#   print(dim(array_X))
#   print(dim(array_Y))
#   #############
#   norm_data<- normalization(X,Y)
#   array_norm_X <- array(t(norm_data[1]), dim = c(inp_win, end_index,dim(trend_train)[2]))
#   array_norm_Y <- array(t(norm_data[2]), dim = c(out_win, end_index,dim(trend_train)[2]))
#   print(dim(array_norm_X))
#   print(dim(array_norm_Y))
#   return(list(X = array_X, Y = array_Y, norm_X = array_norm_X ,norm_Y = array_norm_Y))
#   
# }
############################################3