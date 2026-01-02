#!/bin/bash

PROJECT_ROOT="/home/junspring/mlflow-mnist"

mlflow server \
    --backend-store-uri "sqlite:///${PROJECT_ROOT}/mlflow.db" \
    --default-artifact-root "${PROJECT_ROOT}/mlartifacts" \
    --host 0.0.0.0 \
    --port 5000