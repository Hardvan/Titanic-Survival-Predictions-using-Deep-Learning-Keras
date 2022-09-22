import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Getting Data
train = pd.read_csv('titanic_train.csv')

print(train.head())
print()
print(train.describe())
print()
print(train.info())
print()

# EDA

# Checking Missing Data
fig, axis = plt.subplots()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=axis)

fig, axis = plt.subplots()
sns.countplot(x='Survived', data=train, palette='RdBu_r', ax=axis)

fig, axis = plt.subplots()
sns.countplot(x='Survived', data=train, palette='RdBu_r', hue='Sex', ax=axis)

fig, axis = plt.subplots()
sns.countplot(x='Survived', data=train, palette='rainbow', hue='Pclass', ax=axis)

fig, axis = plt.subplots()
sns.histplot(train['Age'].dropna(), kde=False, color='darkred', bins=30)

fig, axis = plt.subplots()
sns.countplot(x='SibSp',data=train)

fig, axis = plt.subplots()
train['Fare'].hist(color='green',bins=40,figsize=(8,4))

# EDA

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='rainbow')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)

plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

train.drop("PassengerId", axis=1, inplace=True)
train.drop("Cabin", axis=1, inplace=True)

train.dropna(inplace=True)  # For the one Embarked Record

plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Converting Categorical Data to Dummy Variables

# def title_func(name):
    
#     title = name.split()[1]
#     if title in ["Dr.", "Mr.", "Mrs.", "Ms."]:
#         return title
#     else:
#         return "Other"

# def cabin_func(a):
    
#     if a=="NaN":
#         return "N"
#     else:
#         return str(a)[0]

# #train["Title"] = train["Name"].apply(title_func)
# #train["Cabin"] = train["Cabin"].apply(cabin_func)

# title = pd.get_dummies(train["Title"], drop_first=True)
# cabin = pd.get_dummies(train["Cabin"], drop_first=True)
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)

train = pd.concat([train,sex,embark], axis=1)

print(train.head())
print()

# Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), 
                                                    train['Survived'],
                                                    test_size=0.25, 
                                                    random_state=42)

# Scaling Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Shape of X_train:", X_train.shape)

# Early Stopping & Dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(15, activation="relu"))
model.add(Dropout(0.5))

# Binary Classification
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy")

model.fit(x=X_train, y=y_train,
          validation_data=(X_test, y_test),
          epochs=600,
          callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

predictions = (model.predict(X_test) > 0.5)*1

from sklearn.metrics import classification_report, confusion_matrix

print("FOR KERAS:")
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))
print()

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print("FOR Logistic Regression:")
print(confusion_matrix(y_test, predictions))
print()
print(classification_report(y_test, predictions))
print()
















