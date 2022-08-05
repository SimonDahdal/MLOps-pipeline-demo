#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd

from sklearn import tree

import bentoml
from bentoml.io import NumpyNdarray

from sklearn.model_selection import train_test_split


column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]

DATA_FILE = "/home/simon70/Projects/MLOps_demo/Data/abalone.data"

data = pd.read_csv(DATA_FILE, names=column_names)
print("Number of samples: %d" % len(data))



# for more complicated cases use sklearn.feature_extraction.DictVectorizer
for label in "MFI":
    data[label] = data["sex"] == label
del data["sex"]


# In[41]:


data.head()

y = data.rings.values


del data["rings"] # remove rings from data, so we can convert all the dataframe to a numpy 2D array.
X = data.values.astype(np.float)

train_X, test_X, train_y, test_y = train_test_split(X, y) # splits 75%/25% by default


from sklearn.tree import DecisionTreeRegressor
# create an estimator, optionally specifying parameters
model = DecisionTreeRegressor()
# fit the estimator to the data
model.fit(train_X, train_y)
# apply the model to the test and training data
predicted_test_y = model.predict(test_X)
predicted_train_y = model.predict(train_X)


data_percentage_array = np.linspace(10, 100, 10)


train_error = []
test_error = []
for data_percentage in data_percentage_array:
    model = DecisionTreeRegressor(max_depth=10)
    number_of_samples = int(data_percentage / 100. * len(train_y))
    model.fit(train_X[:number_of_samples,:], train_y[:number_of_samples])

    predicted_train_y = model.predict(train_X)
    predicted_test_y = model.predict(test_X)

    train_error.append((predicted_train_y - train_y).std())
    test_error.append((predicted_test_y - test_y).std())


savedmodel = bentoml.sklearn.save_model("abalone_regressor_tree", model)
print(savedmodel)







