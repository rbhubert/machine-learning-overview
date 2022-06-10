# Machine Learning Overview
Projects for the course "2022 Python for Machine Learning and Data Science Masterclass". 

## Data exploration

### [Cleaning _Ames_Housing_Data_ dataset](cleaning/cleaning%20Ames_Housing_Data%20dataset.ipynb) 

We will go through an exploratory data analysis of the dataset that will be used later in the linear regression project. We start by looking at the different features in the dataset and visualizing the values of the outcome variable (SalePrince). We then look for and deal with outliers as well as missing values. This involves understanding the variables and filling in the NaN values accordingly. Finally, we observe and transform the categorical variables to one-hot encoding.

## Supervised algorithms

### [Linear Regression Project](supervised/linear%20regression.ipynb) 

Using the final version of the _Ames_Housing_Data_ dataset, we scaled the features to then create different regression models, namely Basic, Ridge, Lasso, and ElasticNet, to predict the selling price of a house. We perform performance evaluation to select the best models by displaying residuals and a probability plot to help us visualize model performance. Once we select the best models, we tune their hyperparameters using GridSearchCV. Finally, we perform another performance analysis using MAE, RMSE, and residual visualizations.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171257640-cb528fcb-c85f-4864-aa8d-5efe172d3da9.png">
</p>

### [Logistic Regression Project](supervised/logistic%20regression.ipynb) 

For this project, we analyzed the HeartDisease dataset to then create a logistic regression model that will predict whether or not a person has the presence of heart disease based on that person's physical characteristics. We start with an exploratory data analysis, continue with feature scaling, and finally build the model, using cross-validation to find the best hyperparameter. We perform performance analysis using confusion matrix and classification report (accuracy, recall, f1-score, and support) and displays of accuracy recall and ROC curves.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171258375-631c08b9-1e1f-4403-bc19-d665aac735a3.png">
</p>

### [K Nearest Neighbors Project](supervised/k-nearest%20neighbors.ipynb) 

In this project, we explore the Sonar dataset and create a KNN model capable of detecting the difference between a rock and a mine. In this case, we create a Pipeline (which will have the scaler and the KNN model), along with GridSearchCV for tuning the best hyperparameter k. We carry out the performance evaluation through a confusion matrix and a classification report.

<p align="center">
  <img width="768" height="398" src="https://user-images.githubusercontent.com/12433425/171258759-e5ba3cf1-6080-458f-9c68-07d8a7796cec.png">
</p>

### [Support Vector Machine Project](supervised/support%20vector%20machine.ipynb) 

For this project, we performed an analysis of the Winefraude dataset and created an SVM model capable of detecting fraudulent wine samples. As we did in another project, we first performed an exploratory data analysis, scaled the features, and finally built the model using GridSearchCV to find the best C and gamma hyperparameters. Performance analysis was carried out with the confusion matrix and classification report.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171259257-e1542aba-1d86-444e-b1c4-5976dc1a1949.png">
</p>

### [Tree Models Project](supervised/tree%20models.ipynb) 

In this project, we perform an exploratory data analysis of the Telco Customer Churn dataset and create three different tree-based models: decision trees, random forest, and adaboost, to finally compare the performance using the confusion matrix and classification report.

<p align="center">
  <img width="766" height="330" src="https://user-images.githubusercontent.com/12433425/171259274-d3fab9ac-2efc-4e2b-87bb-3bcab3c211e5.png">
</p>

### [Text Classification Project](supervised/text%20classification.ipynb) 

In this project, we create a linear SVC model for the Movie Reviews dataset, using a bag of words and TF-IDF to convert text to numeric vectors. Performance analysis was carried out using confusion matrix and classification_report.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171263931-1a272638-221b-4e1c-9e41-3d32f637dcf5.png"> 
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171263946-01950741-70fa-4dae-8305-b48f8825865b.png">
</p>

## Unsupervised algorithms

### [K-Means Clustering Project](unsupervised/k%20means.ipynb) 

For this project, we will use the CIA Country Facts dataset and create a K-means model to understand the similarity between countries and regions of the world by experimenting with different numbers of groups. The first step was to understand the data, to then prepare the features to be used by the model (dealing with missing values, one-hot coding, scaling the features). The results of the model can be displayed on a world map.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171259360-9e7e073e-9ee3-4850-902d-28050e155c39.png">
</p>

### [Color Quantization Test](unsupervised/color%20quantization%20with%20k-means.ipynb) 

This little project is for testing K-means for color quantization. Basically, we first translate an image into pixels and RGB colors, and then create a K-means model to reduce the amount of color to just k = 10.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171259374-5fab5f7d-d524-481c-a524-143c61ac4422.png">
</p>

### [DBSCAN Project](unsupervised/dbscan.ipynb) 

The DBSCAN project starts by exploring the wholesale customer dataset, then the feature scaling, and finally building the model. The last step is to display the results, with the new values defined by the model.

<p align="center">
  <img width="619" height="411" src="https://user-images.githubusercontent.com/12433425/171259393-cc465420-a674-41e5-80f4-2cc2b9e9c9ae.png">
</p>

### [PCA](unsupervised/pca.ipynb)

For this project, we analyzed the handwritten digit pen-based recognition dataset and used PCA to reduce the number of features that allow a model to identify a number.

<p align="center">
  <img width="762" height="440" src="https://user-images.githubusercontent.com/12433425/171259416-15f5d9ac-304b-4342-a543-75f0dd69f2df.png">
</p>
