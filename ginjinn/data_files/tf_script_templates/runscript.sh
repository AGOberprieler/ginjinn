#!/bin/bash

# Run from your tensorflow/models/research/ directory

export PYTHONPATH=<TF_RESEARCH_PATH>:$PYTHONPATH
export PYTHONPATH=<TF_SLIM_PATH>:$PYTHONPATH

python object_detection/model_main.py \
    --pipeline_config_path=<MODEL_CONFIG_PATH> \
    --model_dir=<MODEL_DIR> \
    --num_train_steps=<N_ITER> \
    --sample_1_of_n_eval_examples=<BATCH_SIZE> \
    <LOG_TO_STDERR>
