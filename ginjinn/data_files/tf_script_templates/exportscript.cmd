:: From tensorflow/models/research/
set PYTHONPATH=<TF_RESEARCH_PATH>;%PYTHONPATH%
set PYTHONPATH=<TF_SLIM_PATH>;%PYTHONPATH%

python object_detection/export_inference_graph.py ^
--input_type=<INPUT_TYPE> ^
--pipeline_config_path=<MODEL_CONFIG_PATH> ^
--trained_checkpoint_prefix=<MODEL_CHECKPOINT_PREFIX> ^
--output_directory=<EXPORT_DIR>
