library(caret)
library(randomForest)
library(glmnet)
library(xgboost)
library(hydroGOF)

setwd("/home/nico/Dokumente/Entwicklung/Uni/PET/repo")

source("load_dataset.R")
source("classification_scores.R")

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

#---------------least squares
lm <- lm(age ~ ., data = train.full)
summary(lm)

lm.test.pred <- predict.lm(lm, test)
lm.RMSE.test <- rmse(lm.test.pred, test$age)
lm.MAE.test <- mae(lm.test.pred, test$age)
lm.class_MAE.test <-  mean(per_class_ME(lm.test.pred,test$age)$ME)

#-----threshold classifier
thresh.MAE.test=mean(abs(test$age-mean(train.full$age)))

#-----------------lasso cross validation-------
# lambda <- c(0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5,10,20,30)
lambda <- seq(0.01, 0.08, by = 0.001)
#lambda=c(10000)
cctrl1 <- trainControl(method = "cv", number = 10)
lasso <- train(age ~ .,
  data = train.full, method = "glmnet",
  trControl = cctrl1,
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = lambda
  )
)

lasso.test.pred <- predict(lasso, test)
lasso.MAE.test <- mae(lasso.test.pred, test$age)
lasso.class_MAE.test <- mean(per_class_ME(lasso.test.pred,test$age)$ME)

coef(lasso$finalModel, lasso$bestTune$lambda)

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
boost.class_MAE.test <-  mean(per_class_ME(boost.test.pred,test$age)$ME)
boost.class_MAE.test_range <-  mean(per_class_ME(boost.test.pred,test$age,5,65)$ME)

View(cbind(boost.test.pred, test$age))

#----------per class error

model <- model.boost
age_pred <- boost.test.pred
age_actual <- test$age

age_me <- per_class_ME(age_pred, age_actual)
avg_per_cl_error <- mean(age_me$ME)

plot(age_me$Age, age_me$ME, axes = FALSE)
ylabel <- seq(0, 100, by = 0.5)
axis(1)
axis(2, at = ylabel, las = 1)
box()
