#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python ./ros/test_model_with_segmentation.py --batch_size 100