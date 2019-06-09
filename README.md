# Semester Project: Age prediction from social networks
## Project structure:
The project is organized into packages. Packages represent various independent tasks that we undertake during the entirity of the project. For example to pre-process data we have the preprocessors package. All the methods to extract features, clean features and create a processed dataset go in the `preprocessor` package.

## Packages
### `./preprocessors`:
Contains methods/classes to implement preprocessing of pokec dataset. Each preprocessing method should have its own file, that defines a method to extract the particular feature from one row of the pokec dataset. Output should be a cleaned and usable represantation of the feature that will directly go into the classifiers/translators etc.