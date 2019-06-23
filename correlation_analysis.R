emb.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/pokec_rel.emb"
pokec.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"

#---------preparation-----------------
setwd("/home/nico/Dokumente/Entwicklung/Uni/PET/repo")
source("load_dataset.R")

d.full.filt=load_dataset(pokec.path,emb.path,TRUE)
d.full.filt=d.full.filt[sample(nrow(d.full.filt), 10000), ]

emb_sample=read.csv(file=emb.path, header=FALSE, sep=" ", nrows = 1)

p=c(c(1),(ncol(emb_sample)+1):ncol(d.full.filt))
d.filt=d.full.filt[,p]

p=c(2:ncol(emb_sample),ncol(d.full.filt))
emb.filt=d.full.filt[,p]

#------------analyze-------------
hist(d.filt$age, breaks=max(d.filt$age))

#-----------correlation-------------
is.quant=sapply(d.filt, Negate(is.factor))
d.quant=d.filt[,is.quant]
#cor(d.filt, method = c("pearson", "kendall", "spearman"))
cor(d.quant, method = c("pearson"))

cols=names(d.filt)
r.sq.list <- list()  
for (i in 1:ncol(d.filt)){
  f=lm(d.filt[,ncol(d.filt)] ~ d.filt[,i], data=d.filt)
  r.sq.list[[i]] <- c(cols[i],summary(f)$adj.r.squared)
}

emb.model=lm(age~., data=emb.filt)
summary(emb.model)

full.model=lm(age~., data=d.full.filt)
summary(full.model)
#------------plot-----------
plot(d.filt)

