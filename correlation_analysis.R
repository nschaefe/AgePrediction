path="/home/nico/Dokumente/Entwicklung/Uni/PET/data/out"
data = read.csv(file=path, header=TRUE, sep="\t")
row.hasNull=
data.filtered = data[!row.hasNull,]
