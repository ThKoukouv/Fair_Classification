## Code for our paper "Interpretable algorithmic fairness in structured and unstructured data"
Our method changes the training of a classifier by adding binary variables ($\boldsymbol{z}$) to encode label flips. During a batch update at each epoch, we update both the model parameters $\boldsymbol{\theta}$ and the binary variables $\boldsymbol{z}$. At the end of an epoch we solve a MIO problem to project to the set of binary variables. 

## Structured data tutorial
In the notebook Structured_Data_Example.ipynb we include a tutorial of our method on the LSAC dataset for tabular data classification. The folder lsac within the folder Data contains the train and test data. The function epoch_manual_str(train_loader, model, z0_train, X_train_moments, X_train, y_train, sens_train, flag) performs an epoch for structured data and takes the following mandatory inputs: 
1) A Pytorch data loader containing the training data (train_loader)
2) The classification model (model)
3) A tensor of binary variables $\boldsymbol{z}$ (z0_train)
4) The moments of the merit attributes (X_train_moments)
5) The data (X_train)
6) The target labels (y_train)
7) The sensitive attribute labels (sens_train)
8) A binary indicator (flag), that is true if class 1 is the privileged class and false otherwise
   
It performs an update for both the model parameters $\boldsymbol{\theta}$ as well as the binary variables $\boldsymbol{z}$ and returns the error and the new value for $\boldsymbol{z}$. In order to run the notebook the following parameters need to be specified:
1) The total number of epochs (num_epochs)
2) The learning rate for the model parameters (lr_theta)
3) The learning rate for the binary variables (lr_z)
4) The tolerance $\epsilon$
   
We note that in order to apply the meritocracy constraints, the parameter merit should be set to true and the meritocracy tolerance $\delta$ should be specified.

## Unstructured data turorial 
In the notebook Unstuctured_Data_Example.ipynb we include a tutorial of our method on the LFW dataset for image classification. When running the notebook the dataset is downloaded from Pytorch and stored in a folder called data. The function epoch_manual_unstr(train_loader, model, z0_train, sens_train, y_train, flag) takes the following mandatory inputs: 
1) A Pytorch data loader containing the training data (train_loader)
2) The classification model (model)
3) A tensor of binary variables $\boldsymbol{z}$ (z0_train)
4) The sensitive attribute labels (s_train)
5) The target labels (y_train)
6) A binary indicator (flag), that is true if class 1 is the privileged class and false otherwise
   
It performs an update for both the model parameters $\boldsymbol{\theta}$ as well as the binary variables $\boldsymbol{z}$ and returns the error and the new value for $\boldsymbol{z}$. In order to run the notebook the following parameters need to be specified:
1) The target attribute
2) The sensitive attribute
3) The total number of epochs (num_epochs)
4) The learning rate for the model parameters (lr_theta)
5) The learning rate for the binary variables (lr_z)
5) The tolerance $\epsilon$

