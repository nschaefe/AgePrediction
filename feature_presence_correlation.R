loadPackage <- function(pkg){
  if(!require(pkg, character.only = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

loadPackage("dplyr")

path="data/out"
df=read.csv(file=path, header=TRUE, sep="\t")

featurePresenceCorr <- function(dat){
  x <- data.frame(sapply(dat %>% select(1:11), Negate(is.na)))
  x$age <- dat$age
  corrs <- sapply((x%>% select(1:11)), cor, y=x$age, use="complete.obs")
  
  return(corrs)
}

y <- featurePresenceCorr(df)
