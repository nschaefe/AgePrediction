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
d2=d
sapply(d, class)
d$user_id=NULL

d.equalized=undersample(d,1500)

d=d.equalized
is_train <- createDataPartition(d$age, p=0.9,list=FALSE)
train <- d[ is_train,]
test  <- d[-is_train,]

is_valid <- createDataPartition(train$age, p=0.1,list=FALSE)
valid <- train[ is_valid,]
train <- train[ -is_valid,]
hist(train$age)
hist(valid$age)

#trainctrl <- trainControl(method = "repeatedcv", number = 10)
#trainctrl <- trainControl(method = "none")
#rpart_tree <- train(age~ ., data = train, method = "rpart", trControl = trainctrl)
#rf_tree <- train(age ~ ., data = train, method = "parRF",trControl = trainctrl)

#important hyperparameters: importance, ntree
train.matrix <- model.matrix(age ~., data=train)
valid.matrix <- model.matrix(age ~., data=valid)

lambda = 0.0001
lasso= glmnet(train.matrix,as.matrix(train$age), alpha =1, lambda =lambda)

rf = randomForest(age~.,data=train, ntree = 200, importance = TRUE)

boost = xgboost(data = train.matrix, label =train$age, 
               eta = 0.1,
               max_depth = 15, 
               nround=500  ,
               objective = "reg:linear",
               nthread = 4
)

model=boost

age_pred=predict(model,valid.matrix)
hist(age_pred)
hist(valid$age)

RMSE.valid=rmse(as.matrix(age_pred),as.matrix(valid$age))
ME.valid=me(as.matrix(age_pred),as.matrix(valid$age))

age_pred=predict(model,test)
MSE.test=rmse(age_pred,as.matrix(test$age))
