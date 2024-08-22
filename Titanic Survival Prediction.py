## CODSOFT DATA SCIENCE INTERNSHIP

## TASK 1: TITANIC SURVIVAL PREDICTION


import pandas as pd
import numpy as np
import sys  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


## Load the Titanic dataset


data = pd.read_csv('Titanic-Dataset.csv')


## Drop unnecessary columns



data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)


## Handle missing values



data['Age'].fillna(data['Age'].median(), inplace=True)


 ## Encode categorical variables



label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])


 ## Replace infinite values with NaN 



data.replace([np.inf, -np.inf], np.nan, inplace=True)


 ## Drop rows with NaN values



data.dropna(inplace=True)


 ## Split the data into features and target



X = data.drop('Survived', axis=1)
y = data['Survived']


 ## Split the data into training and testing sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Train a decision tree classifier



clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


## Make predictions on the test set




y_pred = clf.predict(X_test)


## Evaluate the model



accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

##Print the accuracy


print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)


# ## Thank you!




