library(data.table)
load_dataset = function(pokec.path,emb.path, with_rand_feature=FALSE) {
  id_col="user_id"
  
  pokec = fread(pokec.path, header = T, sep = '\t', data.table = F)
  #read.csv(file=pokec.path, header=TRUE, sep="\t")
  pokec = pokec[sample(nrow(pokec)),]
  
  rand_col=sample.int(100,size=nrow(pokec), replace=TRUE)
  features=cbind(rand_col, pokec)
  
  is_qualitative=c(0,0,1,0,1,0,0,0,0,1,1,1,0)
  # make qualitative vars factor
  for (i in 1:ncol(features)){
    if (is_qualitative[i]){
      features[,i]=factor(features[,i])
    }
  }
  
  emb =fread(emb.path, header = F, sep = ' ', data.table = F)
  #read.csv(file=emb.path, header=FALSE, sep=" ")
  colnames(emb)[1] = id_col
  
  d.full = merge(x = emb, y = features, by = id_col, all = TRUE)
  d.full.filt=d.full[complete.cases(d.full),]
  
  if (!with_rand_feature)
    d.full.filt$rand_col=NULL
  
  return(d.full.filt)
}