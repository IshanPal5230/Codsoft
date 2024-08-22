## CODSOFT DATA SCIENCE INTERNSHIP 

## TASK 5: CREDIT CARD FRAUD DETECTION

## BY Ishan Pal


import pandas as pd


## Load the dataset


data = pd.read_csv('creditcard.csv')


##Explore the Data


print(data.head())
print(data.info())
print(data.describe())


data.isnull().sum()
## Drop or fill missing values as appropriate
data = data.dropna()  ## or data.fillna(value)


from sklearn.preprocessing import StandardScaler

## Assuming all columns except 'Class' are features
features = data.drop('Class', axis=1)
labels = data['Class']

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)


##Oversampling


from imblearn.over_sampling import SMOTE

smote = SMOTE()
features_resampled, labels_resampled = smote.fit_resample(features_normalized, labels)


##Undersampling


from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler()
features_resampled, labels_resampled = undersample.fit_resample(features_normalized, labels)


##Split the Dataset


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.3, random_state=42)


##train a Classification Algorithm


from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)


##Random Forests


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)


##Evaluate the Model


from sklearn.metrics import classification_report, confusion_matrix

y_pred_lr = model_lr.predict(X_test)
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


##Random Forests Evaluation


y_pred_rf = model_rf.predict(X_test)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


##Improve Model Performance


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


##Thank you
