path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"
df=read.csv(file=path, header=TRUE, sep="\t")

df[!is.na(df)]=1
df[is.na(df)]=0

corrs=cor(df, method = c("pearson"))
print(corrs)


