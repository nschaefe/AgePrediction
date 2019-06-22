path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"
df=read.csv(file=path, header=TRUE, sep="\t")
hist(df$age)
df.filt=df[complete.cases(df),]
hist(df.filt$age)
