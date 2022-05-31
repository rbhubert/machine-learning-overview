# ml-ds-masterclass
Projects for the course "2022 Python for Machine Learning and Data Science Masterclass". 

## Data exploration

### [Cleaning _Ames_Housing_Data_ dataset](cleaning/cleaning%20Ames_Housing_Data%20dataset.ipynb) 

Exploration of the dataset that will be used later in the linear regression project. It includes the search and treatment of outliers, missing values and categorical variables.

## Supervised algorithms

### [Linear Regression Project](supervised/linear%20regression.ipynb) 

Creation of different regression models for the _Ames_Housing_Data_ dataset: basic, Ridge, Lasso and ElasticNet with tuned hyperparameters using GridSearchCV. 

Performance analysis using MAE, RMSE and visualizations of residuals.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171257640-cb528fcb-c85f-4864-aa8d-5efe172d3da9.png">
</p>


### [Logistic Regression Project](supervised/logistic%20regression.ipynb) 

Analysis of the HeartDisease dataset and creation of a logistic regression model that will predict whether or not a person has presence of heart disease based on physical features of that person. 

Performance analysis using confusion matrix and classification report (precision, recall, f1-score and support) and visualizations of precision recall and ROC curves.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171258375-631c08b9-1e1f-4403-bc19-d665aac735a3.png">
</p>


### [K Nearest Neighbors Project](supervised/k-nearest%20neighbors.ipynb) 

Exploration of the Sonar dataset and creation of a KNN model capable of detecting the difference between a rock or a mine. Use of Pipeline and GridSearchCV for the tuning of the best k hyperparameter

Performance evaluation using confusion matrix and classification report.

<p align="center">
  <img width="768" height="398" src="https://user-images.githubusercontent.com/12433425/171258759-e5ba3cf1-6080-458f-9c68-07d8a7796cec.png">
</p>


### [Support Vector Machine Project](supervised/support%20vector%20machine.ipynb) 

Analysis of the Wine fraud dataset and creation of a SVM model capable of detecting fraudulent wine samples. Use of GridSearchCV for finding the best C and gamma hyperparameters.

Performance analysis using confusion matrix and classification report.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171259257-e1542aba-1d86-444e-b1c4-5976dc1a1949.png">
</p>

### [Tree Models Project](supervised/tree%20models.ipynb) 

Exploration of the Telco Customer Churn dataset, and creation of three different tree-based models: decision trees, random forest and adaboost. 

Performance evaluation using confusion matrix and classification report.

<p align="center">
  <img width="766" height="330" src="https://user-images.githubusercontent.com/12433425/171259274-d3fab9ac-2efc-4e2b-87bb-3bcab3c211e5.png">
</p>

### [Text Classification Project](supervised/text%20classification.ipynb) 

Creation of a linear SVC model for the Movie Reviews dataset, using bag of words and TF-IDF. 

Performance evaluation using confusion matrix and classification_report.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171263931-1a272638-221b-4e1c-9e41-3d32f637dcf5.png"> 
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171263946-01950741-70fa-4dae-8305-b48f8825865b.png">
</p>

## Unsupervised algorithms

### [K-Means Clustering Project](unsupervised/k%20means.ipynb) 

Analysis of the CIA Country Facts dataset and creation of a K-means model to understand the similarity between countries and regions of the world.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171259360-9e7e073e-9ee3-4850-902d-28050e155c39.png">
</p>

### [Color Quantization Test](unsupervised/color%20quantization%20with%20k-means.ipynb) 

Little test to try K-means for color quantization. We translate a image to pixels and rgb colors, and create a K-means model to reduce the number of color to only k=10. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/171259374-5fab5f7d-d524-481c-a524-143c61ac4422.png">
</p>

### [DBSCAN Project](unsupervised/dbscan.ipynb) 

Exploration of the Wholesale customers dataset, creation of a DBSCAN model and visualization of the results.

<p align="center">
  <img width="619" height="411" src="https://user-images.githubusercontent.com/12433425/171259393-cc465420-a674-41e5-80f4-2cc2b9e9c9ae.png">
</p>

### [PCA](unsupervised/pca.ipynb)

Analysis of the Pen-Based Recognition of Handwritten Digits dataset and use of PCA to reduce the number of features.

<p align="center">
  <img width="762" height="440" src="https://user-images.githubusercontent.com/12433425/171259416-15f5d9ac-304b-4342-a543-75f0dd69f2df.png">
</p>
