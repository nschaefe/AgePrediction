library(caret)
library(randomForest)
library(glmnet)
library(xgboost)
library(hydroGOF)


emb.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/pokec_rel.emb"
pokec.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"

setwd("/home/nico/Dokumente/Entwicklung/Uni/PET/repo")
source("load_dataset.R")

set.seed(123)
d=load_dataset(pokec.path,emb.path)
sapply(d, class)
d$user_id=NULL

d.equalized=undersample(d,2000)

d=d.equalized
is_train <- createDataPartition(d$age, p=0.2,list=FALSE)
train <- d[ is_train,]
test  <- d[-is_train,]

is_valid <- createDataPartition(train$age, p=0.1,list=FALSE)
valid <- train[ is_valid,]
train <- train[ -is_valid,]
#hist(train$age)
#hist(valid$age)

#trainctrl <- trainControl(method = "repeatedcv", number = 10)
#trainctrl <- trainControl(method = "none")
#rpart_tree <- train(age~ ., data = train, method = "rpart", trControl = trainctrl)
#rf_tree <- train(age ~ ., data = train, method = "parRF",trControl = trainctrl)

#important hyperparameters: importance, ntree
train.matrix <- model.matrix(age ~., data=train)
valid.matrix <- model.matrix(age ~., data=valid)
hist(train$age)

#--------------------lasso validation set approach-----------
#lambda <- c(0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5,10,20,30)
lambda <- seq(0.01, 0.08,by=0.001)
lambda_grid=cbind(lambda,0) 
for (i in 1:nrow(lambda_grid)) {
  lamb <- lambda_grid[i,1]
  lasso <- glmnet(train.matrix, as.matrix(train$age), alpha =1, lambda = lamb)

  model <- lasso
  age_pred <- predict(model,s=lamb, valid.matrix)
  RMSE.valid <- rmse(as.matrix(age_pred), as.matrix(valid$age))
  lambda_grid[i,2] <- RMSE.valid
}
hyper_res <- lambda_grid[lambda_grid[,2] == min(lambda_grid[,2]),]
t=predict(model, valid.matrix)
t2=as.matrix(valid$age)
ME.valid <- me(as.matrix(age_pred), as.matrix(valid$age))

#-----------------cross validation-------
cctrl1 <- trainControl(method="cv",number=10)
lasso <- train(age~.,data=train, method = "glmnet", 
                             trControl = cctrl1,
                             tuneGrid = expand.grid(alpha = 1,
                                                    lambda = lambda))

lasso
#test_class_cv_model$bestTune
#coef(test_class_cv_model$finalModel,test_class_cv_model$bestTune$lambda)

age_pred <- predict(lasso,newdata=test)
RMSE.valid <- rmse(as.matrix(age_pred), as.matrix(test$age))

#-----boostin validation set approach---------- 
#final: depth 6, eta=0.1
hyp=expand.grid(eta = c(0.001,0.01,0.1,0.5,1,1.5) ,
            max_depth = c(1,2,3,4,5,6,7,8,9,10))

hyp=cbind(0,hyp) 
for (i in 1:nrow(lambda_grid)) {
  eta <- hyp[i,2]
  max_depth <- hyp[i,3]
  
boost <- xgboost(
  data = train.matrix, label = train$age,
  eta = eta,
  max_depth = max_depth,
  nround = 500,
  objective = "reg:linear",
  nthread = 4
)
age_pred <- predict(boost,newdata=valid.matrix)
RMSE.valid <- rmse(as.matrix(age_pred), as.matrix(valid$age))
hyp[i,1]=RMSE.valid
}
write.csv(hyp,"~/boosting_hyp_search.csv")
hyp.1=hyp[1:51,]
hyper_res.boost <- hyp.1[hyp.1[,1] == min(hyp.1[,1]),]

#----------------TODO
rf <- randomForest(age ~ ., data = train, ntree = 200, importance = TRUE)

hist(age_pred)
hist(valid$age)
ME.valid <- me(as.matrix(age_pred), as.matrix(valid$age))

age_pred <- predict(model, test)
MSE.test <- rmse(age_pred, as.matrix(test$age))

