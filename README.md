# What Reveals Your Exact Age in Social Networks?

From the raw information like profile features and friendship relations we evaluate which features, if revealed, pose a risk to user privacy in the context of age. We perform experiments on a real world online social network and predict the age of users with an average error of 2.62 years, 36\% better than previous work.
The full project report ![here](./project.pdf).

## Structure

 - fcnn/
contains the code for training the FCNN

- preprocessors/
code for parsing and extracting features from the pokec profile data

- load_dataset.R
code to combine graph embedding features and pokec profile data. 
code for preparing the dataset (e.g. outliers, normalization)

- prediction.R
models for predicting age and evaluation 

 - per_class_metric.R
MPA_MAE (see report)

- correlation_analysis.R
basic correlation and R^2 

- outliers.R
experiments to identify outliers

--------------------------
What is not contained:
- histogram generation (R gives this in one line)
- backward selection (we used the code from prediction with a loop removing each feature, one after the other, one at a time)
- node2vec code, see external code section

--------------------------
Remarks:
Most of the files contain one or several path variables which refer to the location of the data. These must be adjusted accordingly.

--------------------------
Dataset:
https://snap.stanford.edu/data/soc-Pokec.html

External Code:
https://github.com/snap-stanford/snap/tree/master/examples/node2vec
we just applied the code as it is to the relationship graph

