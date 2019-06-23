library(caret)
library(randomForest)

emb.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/pokec_rel.emb"
pokec.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"

setwd("/home/nico/Dokumente/Entwicklung/Uni/PET/repo")
source("load_dataset.R")

set.seed(123)
d=load_dataset(pokec.path,emb.path,FALSE)

is_train <- createDataPartition(d$age, p=0.1,list=FALSE)
train <- d[ is_train,]
test  <- d[-is_train,]

is_valid <- createDataPartition(train$age, p=0.1,list=FALSE)
valid <- train[ is_valid,]
train <- train[ -is_valid,]

#trainctrl <- trainControl(method = "repeatedcv", number = 10)
trainctrl <- trainControl(method = "none")
rpart_tree <- train(age~ ., data = train, method = "rpart", trControl = trainctrl)
rf_tree <- train(age ~ ., data = train, method = "parRF",trControl = trainctrl)
rf = randomForest(age~.,data=train,ntree=100)

myGrid <- expand.grid(n.trees = c(150, 175, 200, 225),
                      interaction.depth = c(5, 6, 7, 8, 9),
                      shrinkage = c(0.075, 0.1, 0.125, 0.15, 0.2),
                      n.minobsinnode = c(7, 10, 12, 15))

gbm_tree_tune <- train(age ~ ., data = d, method = "gbm", distribution = "gaussian",
                       trControl = treectrl, verbose = FALSE, tuneGrid = myGrid)


#randomForest(age~ ., data = train, ntree = 500, mtry = 6, importance = TRUE)

