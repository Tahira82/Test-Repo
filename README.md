# Diabetic Prediction
 A project on predicting whether a person is diabetic or not. This project basically makes use of the ensemble technique. An extreme gradient boosting algorithm performs
 well as compared to the basic regression algorithms as per the ensemble learning methodology.  
 Effective with large data sets like this having 768 records.
 Tree algorithms such as XGBoost and Random Forest do not need normalized features.

Futhermore, I preformed the hyperparameter tuning to evaluate the best learning rate,max_depth and n_estimators for the model.

# Dataset link
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

# Website link

https://diabetes-model1.herokuapp.com/


# Tech Stack
* Front-End: HTML, CSS
* Back-End: Flask
* IDE: Pycharm

# How to run this app
* First create a virtual environment by using this command:
* conda create -n myenv python=3.7
* Activate the environment using the below command:
* conda activate myenv
* Then install all the packages by using the following command
* pip install -r requirements.txt
* Now for the final step. Run the app
* python app.py
