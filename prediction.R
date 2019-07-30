library(caret)
library(randomForest)
library(glmnet)
library(xgboost)
library(hydroGOF)

setwd("/home/nico/Dokumente/Entwicklung/Uni/PET/repo")

source("load_dataset.R")
source("per_class_metric.R")

emb.path <- "../data/pokec_rel.emb"
pokec.path <- "../data/out.csv"

set.seed(123)
d <- load_dataset(pokec.path, emb.path)
norm_feat <- c(3, 5, 6, 7, 8) # indices of columns
d <- prepare_dataset(d, norm_feat)
d$user_id <- NULL

is_train=d$is_train
d$is_train=NULL
train <- d[is_train == 1, ]
test <- d[is_train == 0, ]


# build validation set from training set
is_valid <- createDataPartition(train$age, p = 0.1, list = FALSE)
valid <- train[ is_valid, ]
train.full=train
train <- train[ -is_valid, ]

#train=undersample(train,1000)

# to matrix
train.matrix <- model.matrix(age ~ ., data = train)
valid.matrix <- model.matrix(age ~ ., data = valid)
test.matrix <- model.matrix(age ~ ., data = test)

plot_cl_age_pred = function(age_me, initial=TRUE, col="black") {
  if(initial){  
  plot(age_me$Age, age_me$ME, axes = FALSE, xlab = "Age", ylab = "MAE", type = "l", col=col)
  ylabel <- seq(0, 100, by = 2)
  xlabel <- seq(0, 115, by = 2)
  axis(1,at = xlabel,las = 1)
  axis(2, at = ylabel, las = 1)
  box()
  }
  else{
    lines(age_me$Age, age_me$ME, col=col) 
  }
  
}

#-----constant classifier-------
th_pred=rep(mean(train.full$age), length(test$age))
thresh.MAE.test=mae(test$age,th_pred)
thresh.RMSE=rmse(test$age,th_pred)
thresh.class_MAE.test=per_class_ME(th_pred,test$age)
thresh.mean_class_MAE.test <- mean(thresh.class_MAE.test$ME)

plot_cl_age_pred(thresh.class_MAE.test)

#---------------least squares
lm <- lm(age ~ ., data = train.full)
summary(lm)

lm.test.pred <- predict.lm(lm, test)
lm.RMSE.test <- rmse(lm.test.pred, test$age)
lm.MAE.test <- mae(lm.test.pred, test$age)
lm.class_MAE.test=per_class_ME(lm.test.pred,test$age)
lm.mean_class_MAE.test <-  mean(lm.class_MAE.test$ME)

plot_cl_age_pred(lm.class_MAE.test,FALSE, "red")

#-----boosting validation set approach---------- 

#search <- expand.grid(
#  eta = c(0.001, 0.01, 0.1, 0.5, 1, 1.5),
#  max_depth = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
#  nround = c(300,400, 500,600 )
#)

search <- expand.grid(eta = c(0.1), max_depth = c(5), nround = c(400))

model_list <- list()
error <- 0
hyp <- cbind(error, search)
for (i in 1:nrow(hyp)) {
  eta <- hyp[i, 2]
  max_depth <- hyp[i, 3]
  nround <- hyp[i, 4]

  boost <- xgboost(
    data = train.matrix, label = train$age,
    eta = eta,
    max_depth = max_depth,
    nround = nround,
    objective = "reg:linear",
    nthread = 4
  )
  age_pred <- predict(boost, newdata = valid.matrix)
  RMSE.valid <- rmse(as.matrix(age_pred), as.matrix(valid$age))
  ME.valid <- me(as.matrix(age_pred), as.matrix(valid$age))
  hyp[i, 1] <- RMSE.valid
  model_list[[i]] <- boost
}
best_index <- hyp[, 1] == min(hyp[, 1])
hyper_res.boost <- hyp[best_index, ]
model.boost <- model_list[best_index]

boost.test.pred <- unlist(predict(model.boost, newdata = test.matrix))
boost.RMSE.test <- rmse(as.matrix(boost.test.pred), as.matrix(test$age))
boost.MAE.test <- mean(abs(as.matrix(boost.test.pred) - as.matrix(test$age)))
boost.class_MAE_range.test=per_class_ME(boost.test.pred,test$age,5,65)
boost.class_MAE.test=per_class_ME(boost.test.pred,test$age)
boost.mean_class_MAE.test <-  mean(boost.class_MAE.test$ME)
boost.mean_class_MAE_range.test <-  mean(boost.class_MAE_range.test$ME)

plot_cl_age_pred(boost.class_MAE.test,FALSE,"blue")

