setwd("/home/nico/Dokumente/Entwicklung/Uni/PET/repo")
emb.path="../data/pokec_rel.emb"
pokec.path="../data/out.csv"

#---------preparation-----------------
source("load_dataset.R")

d.full.filt=load_dataset(pokec.path,emb.path)
norm_feat <- c(3,5, 6, 7, 8)
d.full.filt=prepare_dataset(d.full.filt,norm_feat) 
d.full.filt$is_train=NULL


emb_sample=read.csv(file=emb.path, header=FALSE, sep=" ", nrows = 1)

pokec_cols_end=ncol(d.full.filt)-ncol(emb_sample)
p=c(1:(pokec_cols_end+1))
d.filt=d.full.filt[,p]

p=c((pokec_cols_end+1):(ncol(d.full.filt)))
emb.filt=d.full.filt[,p]

#-----------correlation-------------
is.quant=sapply(d.filt, Negate(is.factor))
d.quant=d.filt[,is.quant]
cor(d.quant, method = c("pearson"))

#------------------R2----------------
cols=names(d.filt)
r.sq.list <- list()  
for (i in 1:ncol(d.filt)){
  f=lm(age ~ d.filt[,i], data=d.filt)
  r.sq.list[[i]] <- c(cols[i],(summary(f)$r.squared))
}
r.sq.list

emb.model=lm(age~., data=emb.filt)
summary(emb.model)

full.model=lm(age~., data=d.full.filt)
summary(full.model)

