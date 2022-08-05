import numpy as np

from sklearn import tree

import bentoml
from bentoml.io import NumpyNdarray, PandasDataFrame


# In[16]:


BENTO_MODEL_TAG = "sklearn_model:0001"

# In[20]:


regressor = bentoml.sklearn.get("abalone_regressor_tree:latest").to_runner()

service = bentoml.Service(
    "abalone_regressor_tree", runners=[regressor]
)

# Create an API function 
# implementing diffirent endpoints

@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(nparray: np.ndarray) -> np.ndarray:

    # Predict
    result = regressor.run(nparray)
    return np.array(result)


# In[ ]:




