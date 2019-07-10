library(data.table)
library(caret)
load_dataset <- function(pokec.path, emb.path, with_rand_feature = FALSE) {
  id_col <- "user_id"

  colClasses <- c(height = "integer", weight = "integer", smoking = "factor", martial = "factor", gender = "factor", comp_edu = "factor")

  pokec <- fread(pokec.path, header = T, sep = "	", data.table = F, na.strings = "NA", colClasses = colClasses)
  pokec <- pokec[sample(nrow(pokec)), ]

  rand_col <- sample.int(100, size = nrow(pokec), replace = TRUE)
  features <- cbind(rand_col, pokec)

  emb <- fread(emb.path, header = F, sep = " ", data.table = F, na.strings = "NA")
  colnames(emb)[1] <- id_col

  d.full <- merge(x = features, y = emb, by = id_col, all = TRUE)

  if (!with_rand_feature) {
    d.full$rand_col <- NULL
  }
  return(d.full)
}

normalize <- function(d, feat_to_be_norm) {
  train <- d[d$is_train==1,]
  for (i in feat_to_be_norm) {
    feat <- train[, i]
    max_v=max(feat)
    min_v=min(feat)
    # mean=mean(feat)
    # std=std(feat)
    d[, i] <- (d[, i] - min_v) / (max_v - min_v)
  }
  return(d)
}

prepare_dataset <- function(d,norm_feat, as_matrix = FALSE) {
  
  # remove_outlier
  d$height <- replace(d$height, d$height > 250, NA)
  d$weight <- replace(d$weight, d$weight > 500, NA)
  # d$age <- replace(d$age, df$age < 7, NA)
  
  d <- d[complete.cases(d), ]
  
  train_inx <- createDataPartition(d$age, p = 0.8, list = FALSE)
  is_train=integer(nrow(d))
  is_train[train_inx]=1
  d=cbind(d,is_train)
  
  d <- normalize(d, norm_feat)
  
  if (as_matrix) {
    d.matrix <- model.matrix(age ~ ., data = d)
    return(d.matrix)
  }
  return(d)
}

undersample <- function(d, size_per_age) {
  d.equalized <- d
  h <- hist(d.equalized$age, breaks = (max(d.equalized$age)))
  # mean_bin_count=mean(h$counts)
  # can be adjusted to hit tgt set size
  mean_bin_count <- size_per_age
  counts <- c(0, h$counts[1:length(h$counts)])
  diffs <- pmax(0, counts - mean_bin_count)


  for (age_i in 1:length(diffs)) {
    if (diffs[age_i] == 0) {
      next
    }

    index <- seq.int(nrow(d.equalized))
    age <- cbind(index, d.equalized$age)

    rows_with_age <- age[age[, 2] == age_i, ]
    a <- 1:diffs[age_i]
    del_selection <- rows_with_age[a, 1]
    d.equalized <- d.equalized[-del_selection, ]
  }
  return(d.equalized)
}


#emb.path <- "/home/nico/Dokumente/Entwicklung/Uni/PET/data/pokec_rel.emb"
#pokec.path <- "/home/nico/Dokumente/Entwicklung/Uni/PET/data/out.csv"
#d <- load_dataset(pokec.path, emb.path)
#norm_feat <- c(3,5,6,7,8) # indices of columns
#d.pr=prepare_dataset(d,norm_feat)
#d.pr.mat=prepare_dataset(d,norm_feat,as_matrix = TRUE)
#write.csv(d.pr.mat,file="~/data_mat.csv")
#write.csv(d.pr,file="~/data.csv")
