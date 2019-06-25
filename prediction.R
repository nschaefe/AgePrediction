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


#-------------------undersampling---------------
d.equalized=d

h=hist(d$age, bins=max(d$age))
h$breaks
mean_bin_count=mean(h$counts) # can be adjusted to hit tgt set size
diffs=max(0,h$counts-mean_bin_count)

for (age_i in h$breaks){
age=d.equalized$age
age$index=seq.int(nrow(age))

rows_with_age=age[age$age==age_i,]
del_selection=rows_with_age[1:diffs[age_i]]
d.equalized=d.equalized[-del_selection,]
}
#-----------------------------------------------


is_train <- createDataPartition(d$age, p=0.1,list=FALSE)
train <- d[ is_train,]
test  <- d[-is_train,]

is_valid <- createDataPartition(train$age, p=0.1,list=FALSE)
valid <- train[ is_valid,]
train <- train[ -is_valid,]

#trainctrl <- trainControl(method = "repeatedcv", number = 10)
#trainctrl <- trainControl(method = "none")
#rpart_tree <- train(age~ ., data = train, method = "rpart", trControl = trainctrl)
#rf_tree <- train(age ~ ., data = train, method = "parRF",trControl = trainctrl)

#important hyperparameters: importance, ntree
train.matrix <- model.matrix(age ~., data=train)
valid.matrix <- model.matrix(age ~., data=valid)

# lambda_grid= seq(0.01, 0.1 , by=0.01)
lambda_grid <- c(0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5)
lambda_grid$res <- 0
for (i in 1:nrow(lambda_grid)) {
  lamb <- lambda_grid[i]
  lasso <- glmnet(train.matrix, as.matrix(train$age), alpha = 1, lambda = lamb)

  model <- lasso

  age_pred <- predict(model, valid.matrix)
  RMSE.valid <- rmse(as.matrix(age_pred), as.matrix(valid$age))
  lambda_grid$res[i] <- RMSE.valid
}
hyper_res <- lambda_grid[lambda_grid$res == min(lambda_grid$res)]

#----------------------- TODO 
rf <- randomForest(age ~ ., data = train, ntree = 200, importance = TRUE)

boost <- xgboost(
  data = train.matrix, label = train$age,
  eta = 0.1,
  max_depth = 15,
  nround = 500,
  objective = "reg:linear",
  nthread = 4
)


hist(age_pred)
hist(valid$age)
ME.valid <- me(as.matrix(age_pred), as.matrix(valid$age))

age_pred <- predict(model, test)
MSE.test <- rmse(age_pred, as.matrix(test$age))

