path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"
d= read.csv(file=path, header=TRUE, sep="\t")

d.filt=d[complete.cases(d),]
d.filt=d.filt[sample(nrow(d.filt), 10000), ]
hist(d.filt$age, breaks=max(d.filt$age))

# make qualitative vars factor
is_qualitative=c(0,1,0,1,0,0,0,0,1,1,1,0)
for (i in 1:ncol(d.filt)){
  if (is_qualitative[i]){
    d.filt[,i]=factor(d.filt[,i])
  }
}

#cor(d.filt, method = c("pearson", "kendall", "spearman"))
cor(d.filt[,!is_qualitative], method = c("pearson"))

f=lm(age ~ martial, data=d.filt)
summary(f)
predictions = predict(f, d.filt)
#mean((predictions-d.filt$age)^2)

plot(d.filt.quant)





