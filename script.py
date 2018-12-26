# import requisite libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# load training dataset
df = pd.read_csv('train.csv')

# Exploratory data analysis

# view data description
df.info()
# observation:
# 1) Age has less data (NaN values))

# view data correlation
# sns.pairplot(df, hue='Survived')
# observation:
# 1) PClass and Age are highly correlated

# handle missing values for age column with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# features and labels
feature_cols = ['Pclass', 'Sex', 'Age']
label_cols = ['Survived']

# encode the Sex and Pclass columns
labelencoder_sex = LabelEncoder()
df['Sex'] = labelencoder_sex.fit_transform(df['Sex'])

#create dependent and independent datasets
X = df[feature_cols]
y = df[label_cols]

# prepare testing and training data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# perform machine learning algorithms
'''regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor_accuracy = regressor.score(X_test, y_test)'''
# Observation
# accuracy percetage = 33%

svm_linear_clf = SVC(kernel='linear')
svm_linear_clf.fit(X_train, y_train)
svm_linear_clf_accuracy = svm_linear_clf.score(X_test, y_test)
# Observation
# accuracy percentage 77%

'''svm_poly_clf = SVC(kernel='poly')
svm_poly_clf.fit(X_train, y_train)
svm_poly_clf_accuracy = svm_poly_clf.score(X_test, y_test)

svm_rbf_clf = SVC(kernel='rbf')
svm_rbf_clf.fit(X_train, y_train)
svm_rbf_clf_accuracy = svm_rbf_clf.score(X_test, y_test)'''

# test the model with the shared dataset
testdata = pd.read_csv('test.csv')
testdata.info()

# handle missing values for age column with median
testdata['Age'].fillna(testdata['Age'].median(), inplace=True)

labelencoder_sex_test = LabelEncoder()
testdata['Sex'] = labelencoder_sex_test.fit_transform(testdata['Sex'])

testdata['Survived'] = svm_linear_clf.predict(testdata[feature_cols])

predict = pd.DataFrame(predict)
predict['PassengerId'] = df['PassengerId']
predict.rename(columns={'Survived':'PassengerId'}, inplace=True)
predict.to_csv('Survival_predictions1.csv')



submission = pd.DataFrame({'PassengerId':testdata['PassengerId'],'Survived':testdata['Survived']})
submission.to_csv('Survival_predictions.csv',index=False)
