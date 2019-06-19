path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out"
d= read.csv(file=path, header=TRUE, sep="\t")

#---------preparation-----------------
d.filt=d[complete.cases(d),]
d.filt=d.filt[sample(nrow(d.filt), 2000), ]

rand_col=sample.int(100,size=nrow(d.filt), replace=TRUE)
d.filt=cbind(rand_col,d.filt)

# make qualitative vars factor
is_qualitative=c(0,0,1,0,1,0,0,0,0,1,1,1,0)
for (i in 1:ncol(d.filt)){
  if (is_qualitative[i]){
    d.filt[,i]=factor(d.filt[,i])
  }
}

#------------analyze-------------
hist(d.filt$age, breaks=max(d.filt$age))


#-----------correlation-------------
d.qual=d.filt[,!is_qualitative]
#cor(d.filt, method = c("pearson", "kendall", "spearman"))
cor(d.qual, method = c("pearson"))


r.sq.list <- list()  
for (i in 1:ncol(d.filt)){
    f=lm(d.filt[,ncol(d.filt)] ~ d.filt[,i], data=d.filt)
    r.sq.list[[i]] <- c(n[i],summary(f)$adj.r.squared)
}


#------------plot-----------
plot(d.filt)



