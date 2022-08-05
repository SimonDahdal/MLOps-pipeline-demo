#!/bin/bash

latest_path='/home/simon70/bentoml/bentos/abalone_regressor_tree/latest'

docker tag abalone_regressor_tree:`cat $latest_path` localhost:5001/abalone_regressor_tree:latest
docker push localhost:5001/abalone_regressor_tree:latest
