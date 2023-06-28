#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python ./ros/test_model_with_segmentation.py --refine_steps 25 --num_grasp_samples 80 --batch_size 80 --threshold 0.9 --choose_fn better_than_threshold_in_sequence
