emb.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/pokec_rel.emb"
path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"
id_col="user_id"

#---------preparation-----------------
features = read.csv(file=path, header=TRUE, sep="\t")

rand_col=sample.int(100,size=nrow(features), replace=TRUE)
features=cbind(rand_col,features)

is_qualitative=c(0,0,1,0,1,0,0,0,0,1,1,1,0)
# make qualitative vars factor
for (i in 1:ncol(features)){
  if (is_qualitative[i]){
    features[,i]=factor(features[,i])
  }
}

emb = read.csv(file=emb.path, header=FALSE, sep=" ")
colnames(emb)[1] = id_col

d.full = merge(x = emb, y = features, by = id_col, all = TRUE)

d.full.filt=d.full[complete.cases(d.full),]
d.full.filt=d.full.filt[sample(nrow(d.full.filt), 10000), ]


p=c(c(1),(ncol(emb)+1):ncol(d.full.filt))
d.filt=d.full.filt[,p]

p=c(2:ncol(emb),ncol(d.full.filt))
emb.filt=d.full.filt[,p]

#------------analyze-------------
hist(d.filt$age, breaks=max(d.filt$age))

#-----------correlation-------------

d.qual=d.filt[,!is_qualitative]
#cor(d.filt, method = c("pearson", "kendall", "spearman"))
cor(d.qual, method = c("pearson"))


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

