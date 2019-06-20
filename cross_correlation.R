path="~/Desktop/Saar/ss2019/pets/data/profiles_cleaned.txt"
d= read.csv(file=path, header=TRUE, sep="\t")

#---------preparation-----------------
d.filt=d[complete.cases(d),]
# d.filt=d.filt[sample(nrow(d.filt), 2000), ]

rand_col=sample.int(100,size=nrow(d.filt), replace=TRUE)
d.filt=cbind(rand_col,d.filt)

# make qualitative vars factor
is_qualitative=c(0,0,1,0,1,0,0,0,0,1,1,1,0)
# for (i in 1:ncol(d.filt)){
#   if (is_qualitative[i]){
#     d.filt[,i]=factor(d.filt[,i])
#   }
# }
d.numerical = d.filt[,!as.logical(is_qualitative)]
d.categorical = d.filt[, as.logical(is_qualitative)]
#-----------correlation-------------

corr = cor(d.numerical, method = c("pearson"))

# ------------VIF for multicollinearity --------------------
# Build Linear Model
model1 <- lm(age ~., data = d.numerical)
vif = car::vif(model1)

# ------------Chi2------------------------------------------
# Perform Chi2 test measures correlation between each combination qualitative of variables.
ind<-combn(NCOL(d.categorical),2)
x2 = lapply(1:NCOL(ind), function (i) chisq.test(d.categorical[,ind[1,i]], d.categorical[,ind[2,i]]))
# Add column names of the combination between which chi2 is performed to interpret
# the results
c_names = names(d.categorical)
n = c()
for(i in 1:NCOL(ind)){
  n = c(n, paste( c_names[ind[1,i]]," ,", c_names[ind[2,i]]))
}
names(x2) <- n
# Print something
x2[1]
# ------------Logistic Regression--------------------
# Used as a measure of corellation between a qualitative varible and quantitative variables
# P values of a logistic regression model can be used as a mesure of correlation

logreg_op = lapply(1:NCOL(d.categorical), function (i) summary (glm(d.categorical[, i] ~.,  data=d.numerical)))
n2 = c()
for(i in 1:NCOL(d.categorical)){
  n2 = c(n2, c_names[i])
}
names(logreg_op) = n2
logreg_op[1]
