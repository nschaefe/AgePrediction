requiredPackages = c('sqldf','ggplot2')
for(p in requiredPackages){
  if(!require(p,character.only = TRUE)) install.packages(p)
  library(p,character.only = TRUE)
}


plotHist <- function(dat, caption, xLabel, yLabel){
  ggplot(data=dat, aes(dat$age)) + geom_bar(stat="count") + 
    labs(title = caption, x = xLabel, y = yLabel) +
    theme(axis.text.x = element_text(angle=90, vjust = 0.5))
}


dataWithNA <- function(dat){
  datNoNA <- na.omit(dat)
  datWithNA <- sqldf('SELECT * FROM dat EXCEPT SELECT * FROM datNoNA')
  return(datWithNA)
}

path="data/out"
df=read.csv(file=path, header=TRUE, sep="\t")
df$age <- factor(df$age, exclude = NULL)

# Distribution of age for the dataset
plotHist(df, "Age Distribution", "Age", "Frequency")


# Reloaded df because the age column was converted to a factor
# in order to include missing values in the plot.
# This (age column as a factor) however causes na.omit to produce wrong results. 
# Thus the need to reload the data
df=read.csv(file=path, header=TRUE, sep="\t")
dfNA <- dataWithNA(df)
dfNA$age <- factor(dfNA$age, exclude = NULL)

plotHist(dfNA, "Age distribution for rows with missing values", "Age", "Frequency")