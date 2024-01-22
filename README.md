Code for our paper "Interpretable algorithmic fairness in structured and unstructured data" 

Our method changes the training of a classifier by adding binary variables ($z$) to encode label flips. At the end of each epoch we solve a MIO problem to project to the set of binary variables. 

For structured data we include a tutorial of our approach on the LSAC dataset on Stuctured_Data_Example.ipynb. The folder lsac within the folder Data contains the train and test data. The function epoch_manual_str takes the following mandatory inputs: 
1) A Pytorch data loader containing the training data
2) The classification model
3) A tensor of binary variables $z$
4) The moments of the merit attributes
5) The data (X_train)
6) The target labels (y_train)
7) The sensitive attribute labels (s_train)
8) A binary indicator (flag), that is true if class 1 is the privileged class and false otherwise
It performs an update for both the model parameters and the binary variables and returns the error and the new value for $z$
