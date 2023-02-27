## Data
This folder contains raw data and the scripts that were used for preprocessing.
## Preprocessing
The dataset that was used is cloned from the worl done by [Sharma et al.](https://github.com/tushartushar/DeepLearningSmells/tree/master/data). 

First step in preprocessing is dealing with explosive vocabulary problem, where the number of tokens collected from identifiers, function names and constants can be theoretically infinit. To circumvent this we replace these tokens with abstract names: VAR1, VAR2 ... FUNC1, FUNC2 ect.