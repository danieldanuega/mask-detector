# Coral TPU Transfer Learning

This branch will use transfer learning method to create mask detector. The pre-trained model will use _SSD MobileNet V2 COCO_ provided by Tensorflow Model in Google Coral Website.

## Model

Pre-trained model (SSD MobileNet V2 COCO) that compatible with Google Coral engine https://www.coral.ai/models/.

## Dataset

This repo use the TFRecord dataset (https://drive.google.com/drive/folders/1KYTPRmxJHs8uKTL0exjOSqFhXJtvnk4r) that is provided by Wang Yifan at GitHub. You can check the link at Wang's GitHub https://github.com/wangyifan411/Face-Mask-Type-Detector/issues/1.

## How to

This model is trained by using the tutorial given by Coral team, in this case using docker approach. You can see the tutorial at Coral website https://www.coral.ai/docs/edgetpu/retrain-detection/. Please read carefully to understand whats going on.

1. Download the model and the dataset
2. replace pipeline.config at ssd_mobilexxx with our pipeline.config
3. replace the constants.sh in container with ours

## Train

The training process use the `constants.sh` that come with the docker container and is modified by me for the file locations purpose.

### Note

- I include the tutorial's Dockerfile in case in the future there will be any changes, so I personally could improve or edit the model without any compatibilty issue.
- The pipeline.config that I used for ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03 was from `/tutorials/docker/object_detection/scripts/configs/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config`
