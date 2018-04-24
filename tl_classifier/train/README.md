# Training and Exporting the Classifier

## Requirements

* Tensorflow 1.3.0 (the version used on Udacity's Carla)
* Protobuf 2.6.1 (both the python library and protoc executable)
* tensorflow/models repo commit # `edcd29f2dbb4b3eaed387fe17cb5270f867aec42` installed in the directory above this one (CarND-Capstone/tl-classifier) using the protoc version above

## Training

1. Download [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) and extract the faster_rcnn_resnet101_coco_xxxxxxxxxx folder into this directory.
1. Retreive a labeled dataset and create record files within this directory.
1. From within this directory, type:
    ```cmd
    python train.py --logtostderr --train_dir=./model_ckpts --pipeline_config_path=faster_rcnn_resnet101_coco.config
    ```
    and wait until training completes (it may take a long time to start up).
1. Type:
    ```cmd
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix ./model_ckpts/model.ckpt-10000 --output_directory ./fine_tuned_model
    ```
1. Use `fine_tuned_model/frozen_inference_graph.pb` from Tensorflow for your classifications!
