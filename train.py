#  Doing the necessary imports
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Reading the features and the labels
data = pd.read_csv('pima-indians-diabetes.csv')

# Grabing the initial info of the dataset
print(data.info())

# Count the number of zeros in all columns of Dataframe
for column_name in data.columns:
    count = (data[column_name] == 0).sum()
    print(f'The count of zeros in column {column_name} is {count}')

cols = ['Plasma glucose concentration',
        'Diastolic blood pressure (mm Hg)', 'Triceps skinfold thickness (mm)',
        '2-Hour serum insulin (mu U/ml)',
        'Body mass index (weight in kg/(height in m)^2)',
        'Diabetes pedigree function', 'Age']

# As mentioned in the data description,the missing values have been replaced by zeroes. So, we are replacing zeroes with nan
for col in cols:
    data[col] = data[col].replace(0, np.nan)

# checking for missing values
data.isna().sum()

# Imputing the missing values
data['Plasma glucose concentration'] = data['Plasma glucose concentration'].fillna(data['Plasma glucose concentration'].mode()[0])
data['Diastolic blood pressure (mm Hg)'] = data['Diastolic blood pressure (mm Hg)'].fillna(data['Diastolic blood pressure (mm Hg)'].mode()[0])
data['Triceps skinfold thickness (mm)'] = data['Triceps skinfold thickness (mm)'].fillna(data['Triceps skinfold thickness (mm)'].mean())
data['2-Hour serum insulin (mu U/ml)'] = data['2-Hour serum insulin (mu U/ml)'].fillna(data['2-Hour serum insulin (mu U/ml)'].mean())
data['Body mass index (weight in kg/(height in m)^2)'] = data['Body mass index (weight in kg/(height in m)^2)'].fillna(data['Body mass index (weight in kg/(height in m)^2)'].mean())

# checking for missing values after imputation
data.isna().sum()

# Separating the feature and the Label columns
x = data.drop(labels='Is Diabetic', axis=1)
y = data['Is Diabetic']

# As the datapoints differ a lot in magnitude, we'll scale them
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)
train_x, test_x, train_y, test_y = train_test_split(scaled_data, y, test_size=0.3, random_state=42)

# fit model no training data
model = XGBClassifier(objective='binary:logistic')
model.fit(train_x, train_y)

# Cheking training accuracy
y_pred = model.predict(train_x)
predictions = [round(value) for value in y_pred]
training_accuracy = accuracy_score(train_y, predictions)
print(f"training_accuracy : {training_accuracy}")

# cheking initial test accuracy
y_pred = model.predict(test_x)
predictions = [round(value) for value in y_pred]
test_accuracy = accuracy_score(test_y, predictions)
print(f"test_accuracy: {test_accuracy} ")

# Now to increase the accuracy of the model, we'll perform Hyper-parameter-tuning using Grid Search
param_grid = {

    'learning_rate': [1, 0.5, 0.1, 0.01, 0.001],
    'max_depth': [3, 5, 10, 20],
    'n_estimators': [10, 50, 100, 200]
}
grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid)
grid.fit(train_x, train_y)

# To  find the parameters givingmaximum accuracy
print(f'grid.best_params_ : {grid.best_params_}')

# Create new model using the same parameters
new_model = XGBClassifier(learning_rate=1, max_depth=5, n_estimators=50)
new_model.fit(train_x, train_y)

y_pred_new = new_model.predict(test_x)
predictions_new = [round(value) for value in y_pred_new]
test_accuracy_new = accuracy_score(test_y, predictions_new)
print(f"test_accuracy_new : {test_accuracy_new}")

# As we have increased the accuracy of the model, we'll save this model
filename = 'xgboost_model.pickle1'
pickle.dump(new_model, open(filename, 'wb'))

# Load the saved model for prediction
loaded_model = pickle.load(open(filename, 'rb'))

# We'll save the scaler object as well for prediction
filename_scaler = 'scaler_model.pickle'
pickle.dump(scaler, open(filename_scaler, 'wb'))


