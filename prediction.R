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
is_train <- createDataPartition(d$age, p=0.5,list=FALSE)
train <- d[ is_train,]
test  <- d[-is_train,]

is_valid <- createDataPartition(train$age, p=0.1,list=FALSE)
valid <- train[ is_valid,]
train <- train[ -is_valid,]
#hist(train$age)
#hist(valid$age)

train.matrix <- model.matrix(age ~., data=train)
valid.matrix <- model.matrix(age ~., data=valid)
test.matrix <- model.matrix(age ~., data=test)
hist(train$age)


#-----------------lasso cross validation-------
#lambda <- c(0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5,10,20,30)
lambda <- seq(0.01, 0.08,by=0.001)

cctrl1 <- trainControl(method="cv",number=10)
lasso <- train(age~.,data=train, method = "glmnet", 
                             trControl = cctrl1,
                             tuneGrid = expand.grid(alpha = 1,
                                                    lambda = lambda))


model.lasso=lasso
test.pred= predict(model.lasso,test)
RMSE.test = rmse(test.pred, test$age)
ME.test = mean(abs(test.pred- test$age))

#lasso
#test_class_cv_model$bestTune
#coef(test_class_cv_model$finalModel,test_class_cv_model$bestTune$lambda)


#-----boostin validation set approach---------- 
#final: depth 6, eta=0.1
search=expand.grid(eta = c(0.001,0.01,0.1,0.5,1,1.5) ,
            max_depth = c(1,2,3,4,5,6,7,8,9,10))

search=expand.grid(eta = c(0.1) ,max_depth = c(4,5,6), nround=c(400,500))

model_list=list()
error=0
hyp=cbind(error,search) 
for (i in 1:nrow(hyp)) {
  eta <- hyp[i,2]
  max_depth <- hyp[i,3]
  nround <- hyp[i,4]
  
boost <- xgboost(
  data = train.matrix, label = train$age,
  eta = eta,
  max_depth = max_depth,
  nround = nround,
  objective = "reg:linear",
  nthread = 4
)
age_pred <- predict(boost,newdata=valid.matrix)

RMSE.valid <- rmse(as.matrix(age_pred), as.matrix(valid$age))
ME.valid = me(as.matrix(age_pred), as.matrix(valid$age))
hyp[i,1]=RMSE.valid
model_list[[i]]=boost
}
best_index=hyp[,1] == min(hyp[,1])
hyper_res.boost = hyp[best_index,]
model.boost=model_list[best_index]

test.pred= unlist(predict(model.boost,newdata=test.matrix))
boost.RMSE.test = rmse(as.matrix(test.pred), as.matrix(test$age))
boost.ME.test = mean(abs(as.matrix(test.pred)- as.matrix(test$age)))


View(cbind(test.pred,test$age))

#----------------TODO--------------
rf <- randomForest(age ~ ., data = train, ntree = 200, importance = TRUE)

hist(age_pred)
hist(valid$age)
ME.valid <- me(as.matrix(age_pred), as.matrix(valid$age))

age_pred <- predict(model, test)
MSE.test <- rmse(age_pred, as.matrix(test$age))

