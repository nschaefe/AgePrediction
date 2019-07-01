library(caret)
classification_scores = function(pred,actual,thresh) {
  pred_lab=pred
  ground_lab=  actual
  max(test.pred)
  max(test$age)
  
  pred_clipped=cut(pred_lab,thresh)
  ground_clipped=cut(ground_lab,thresh)
  
  conf_mat=confusionMatrix(pred_clipped,ground_clipped)
  scores=conf_mat$byClass
  return(scores)
}