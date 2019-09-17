:: copy tensorflow/models to conda environment
(robocopy "%SRC_DIR%\models" "%CONDA_PREFIX%\models" /s /e /njh /njs /ndl /nc /ns /nfl) ^& IF %ERRORLEVEL% LSS 8 SET ERRORLEVEL = 0

:: compile protobuf files
START_DIR=%cd%
cd "%CONDA_PREFIX%\models\research"
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\losses.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\calibration.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\eval.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\flexible_grid_anchor_generator.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\ssd.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto
cd "%START_DIR%"

:: copy pycocotools to tensorflow/models/reaserch
(robocopy "%SRC_DIR%\cocoapi\PythonAPI\pycocotools" "%CONDA_PREFIX%\models\research\pycocotools" /s /e /njh /njs /ndl /nc /ns /nfl) ^& IF %ERRORLEVEL% LSS 8 SET ERRORLEVEL = 0