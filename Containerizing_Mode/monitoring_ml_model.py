import numpy as np
from sklearn import tree
import bentoml
from bentoml.io import NumpyNdarray


BENTO_MODEL_TAG = "abalone_regressor_tree:latest"

regressor = bentoml.sklearn.get(BENTO_MODEL_TAG).to_runner()

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





