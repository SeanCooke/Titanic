# titanic

## Results
Achieving 86% accuracy on a test data set using an xgboost classifier to predict whether a passenger survived the Titanic disaster. 86% accuracy is also achieved on the training data set which indicates the model is not overfit to our training data

test set confusion matrix
```
Dead Correctly Classified as Dead:		127
Survivors Incorrectly Classified as Dead:	19
Dead Incorrectly Classified as Survivors:	12
Survivors Correctly Classified as Survivors:	65
```

## xgboost
xgboost uses an ensemble of decision trees. this model was selected as it is assumed that when the titanic was sinking, life boats were given to passengers based on a small number of demographic/familial factors. model tuning for the xgboost model is shown below
``` python
xgb = XGBClassifier(n_estimators=100,
                    max_depth=3,
                    learning_rate=.1,
                    subsample=.9,
                    colsample_bytree=1,
                    gamma=1
)
```

## Feature Engineering
The below features were supplied to the xgboost model
```
Fare - (float) the amount a passenger paid for their ticket
family_bucket_1 - (boolean) 1 if the passenger was traveling alone, 0 otherwise
family_bucket_2 - (boolean) 1 if the passenger was traveling with 1 to 3 other people, 0 otherwise
family_bucket_3 - (boolean) 1 if the passenger was traveling with 4 or more people, 0 otherwise
has_cabin - (boolean) 1 if the passenger has non-null data in the 'Cabin' column, 0 otherwise
is_male - (boolean) 1 if the passenger's Sex is 'male', 0 otherwise
has_age - (boolean) 0 if the passenger's age is np.nan, 1 otherwise
age_bin - (integer) rounds Age up to nearest 5-year interval (becomes 5, 6 becomes 10, etc...). if age is null, age_bin is 40
is_child - (boolean) 1 if Age <= 20, 0 otherwise
class_1 - (boolean) 1 if Pclass is 1, 0 otherwise
class_2 - (boolean) 1 if Pclass is 2, 0 otherwise
class_3 - (boolean) 1 if Pclass is 3, 0 otherwise
```

## Requirements
Built with [conda 4.7.12](https://www.anaconda.com/distribution/), [Python 3.7.3](https://www.python.org/downloads/release/python-373/), [numpy 1.16.2](https://www.scipy.org/install.html), [pandas 0.24.2](https://pypi.org/project/pandas/0.24.2/), [sklearn 0.21.3](https://scikit-learn.org/stable/install.html), and [xgboost 0.90](https://pypi.org/project/xgboost/)
