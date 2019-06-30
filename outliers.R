path="data/out"
df=read.csv(file=path, header=TRUE, sep="\t")


# view summary to check if any obvious errors can be seen
summary(df)
# It can be accurately inferred from summary that the first 6
# columns contain no outlier values
# obvious errors include height > 250cm or height < 30cm, 
# weights > 1000kg or weight < 10kg, age < 7

unique(df$gender)
# It can be further conferred that the gender column contains no
# outlier values by checking for the unique gender values in the dataset

# remove obvious errors
df$height <- replace(df$height, df$height > 250 | df$height < 30, NA)
df$weight <- replace(df$weight, df$weight > 1000 | df$weight < 10, NA)
df$age <- replace(df$age, df$age < 7, NA)

# We can therefore make a boxplot of the last 6 columns to check for outlier
# values and see if it can be determined the occurrence is an obvious error
# not
library("dplyr")
boxplot(df %>% select(7:12))

#From the boxplot, the remaining outliers cannot be concluded to be obvious 
# errors and are therefore not removed from the dataset


### The following columns also contain no outlier values
#unique(df$comp_edu)
#unique(df$marital)
#unique(df$martial)
#unique(df$smoking)

write.csv(df,"data/no_outliers", row.names = FALSE)
