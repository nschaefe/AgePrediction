library(caret)
  
per_class_ME = function(age_pred,age_actual, age_thresh_low=-1,age_thresh_high=Inf) {
  ages=cbind(age_actual,age_pred)
  age_me =(data.frame("Age"=c(0), "ME"=c(0)))
  unique(age_actual)
  for (age in  unique(age_actual)){
    if (age > age_thresh_high || age < age_thresh_low ){
      next
    }
    age_grp= ages[age==ages[,1],]
    if(length(age_grp)==2){
      ME.age = mean(abs(age_grp[1]-age_grp[2]))
    }
    else{
      ME.age = mean(abs(age_grp[,1]-age_grp[,2]))
    }
    age_me=rbind(age_me,c(age,ME.age))
  }
  age_me=age_me[2:nrow(age_me),]
  age_me=age_me[order(age_me$Age),]
 
  return (age_me)
}

