#!/bin/bash

latest_path='$HOME/bentoml/bentos/abalone_regressor_tree/latest'

docker tag abalone_regressor_tree:`cat $latest_path` localhost:5001/abalone_regressor_tree:latest
docker push localhost:5001/abalone_regressor_tree:latest
