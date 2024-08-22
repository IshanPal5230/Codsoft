## CODSOFT DATA SCIENCE INTERNSHIP 

## TASK 3: IRIS FLOWER DETECTION

## BY Ishan Pal


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


 ## Load the Iris flower dataset


df = pd.read_csv('IRIS.csv')


 ## Split the dataset into features (measurements) and labels (species)



X = df.drop('species', axis=1)
y = df['species']


 ## Split the dataset into training and testing sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Train a K-Nearest Neighbors classifier on the training data



classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)


 ## Predict the species for the test data



y_pred = classifier.predict(X_test)


## Calculate accuracy of the model



accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


## Thank you




