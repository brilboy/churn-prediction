# Import libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Load the dataset
data = pd.read_csv(r'C:\Users\Ahmad Jibril H\Documents\Data pribadi\Coding\Projects\customer-churn-prediction\src\data\updated_customer_data.csv')

# Define the features (X) and target variable (y)
X = data.drop('Target', axis=1)
y = data['Target']

# Split the data into training and testing sets (adjust the test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logistic_regression = LogisticRegression()
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['none', 'l2'],
}
grid_search_lr = GridSearchCV(logistic_regression, param_grid=param_grid_lr, cv=5, n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}
grid_search_knn = GridSearchCV(knn, param_grid=param_grid_knn, cv=5, n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_

# Create Gaussian Naïve Bayes model
gnb = GaussianNB()
scores_cv = cross_val_score(gnb, X_train, y_train, cv=5, scoring='accuracy') # Perform cross-validation
gnb.fit(X_train, y_train) # Fit the model to the entire training set

# Evaluate Logistic Regression
y_pred_lr = best_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Evaluate K-Nearest Neighbors
y_pred_knn = best_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\nK-Nearest Neighbors Accuracy:", accuracy_knn)
print("K-Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("K-Nearest Neighbors Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# Evaluate Naïve Bayes model
print("\nGaussian Naïve Bayes Cross-validation:") # Print the cross-validation scores
print("Cross-validation scores:", scores_cv)
print("Mean accuracy:", scores_cv.mean())
y_pred_nb = gnb.predict(X_test) # Make predictions on the test set
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Gaussian Naïve Bayes Accuracy:", accuracy_nb)
print("Classification Report for Gaussian Naïve Bayes:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix for Gaussian Naïve Bayes:")
print(confusion_matrix(y_test, y_pred_nb))

# Save the entire models
joblib.dump(best_lr, r'src\model\logistic_regression_model.pkl') # Save Logistic Regression model
joblib.dump(best_knn, r'src\model\knn_model.pkl') # Save K-Nearest Neighbors model
joblib.dump(scores_cv, r'src\model\gaussian_nb_model.pkl') # Save Gaussian Naïve Bayes model