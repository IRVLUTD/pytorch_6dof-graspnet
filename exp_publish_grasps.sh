#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python ./ros/test_model_with_segmentation.py --refine_steps 50 --num_grasp_samples 60 --batch_size 30 --choose_fn better_than_threshold_in_sequence 
