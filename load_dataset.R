library(data.table)
load_dataset = function(pokec.path,emb.path, with_rand_feature=FALSE) {
  id_col="user_id"
  
  colClasses = c(height = "integer", weight="integer", smoking="factor", martial="factor", gender = "factor", comp_edu="factor")
  
  pokec = fread(pokec.path, header = T, sep = '\t', data.table = F, na.strings="NA",colClasses = colClasses )
  #pokec=read.csv(file=pokec.path, header=TRUE, sep="\t",colClasses = colClasses )
  pokec = pokec[sample(nrow(pokec)),]
  
  rand_col=sample.int(100,size=nrow(pokec), replace=TRUE)
  features=cbind(rand_col, pokec)

  emb =fread(emb.path, header = F, sep = ' ', data.table = F, na.strings="NA")
  #read.csv(file=emb.path, header=FALSE, sep=" ")
  colnames(emb)[1] = id_col
  
  d.full = merge(x = emb, y = features, by = id_col, all = TRUE)
  d.full.filt=d.full[complete.cases(d.full),]
  
  if (!with_rand_feature)
    d.full.filt$rand_col=NULL
  

  return(d.full.filt)
}

undersample = function(d, size_per_age) {
d.equalized=d
h=hist(d.equalized$age, breaks=(max(d.equalized$age)))
#mean_bin_count=mean(h$counts)
# can be adjusted to hit tgt set size
mean_bin_count=size_per_age
counts=c(0,h$counts[1:length(h$counts)])
diffs=pmax(0,counts-mean_bin_count)


for (age_i in 1:length(diffs)){
  if(diffs[age_i]==0){
    next
  }
  
  index=seq.int(nrow(d.equalized))
  age=cbind(index,d.equalized$age)
  
  rows_with_age=age[age[,2]==age_i,]
  a=1:diffs[age_i]
  del_selection=rows_with_age[a,1]
  d.equalized=d.equalized[-del_selection,]
  
}
return(d.equalized)

}


#emb.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/pokec_rel.emb"
#pokec.path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"
#load_dataset(pokec.path,emb.path)