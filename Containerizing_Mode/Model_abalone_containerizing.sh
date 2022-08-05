#!/bin/bash

source /home/simon70/Notebooks/py-envs/venv1/bin/activate
bentoml build
bentoml containerize abalone_regressor_tree:latest


