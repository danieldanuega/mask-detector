import numpy as np
import tensorflow as tf
from imutils import paths
import cv2
from subprocess import PIPE, Popen

"""
This is the quantization script for tflite model float32 to int8 model.
I use the original dataset by Wang Yifan for calibration purpose https://drive.google.com/drive/folders/1w-zoOuSauw5xBhOrfZpjcglvDH14mlGu.

The dataset calibration is not included in this repo, please see the issue pages of Wang Yifan for further information. 
https://github.com/wangyifan411/Face-Mask-Type-Detector/issues/1

FYI:
the model that is used by Wang Yifan is SSD_Mobilenet_V2_COCO, where the size is 300x300.
You could find more information about size, batch_size, etc, of the pre-trained model in tensorflow object detection model zoo 1.
This repo use Tensorflow 1
"""

# normH, normW = (300, 300)
# BATCH_SIZE = 24

script = [
    "tflite_convert",
    "--output_file=./TFLite_model/detect_model_quant.tflite",
    "--graph_def_file=./TFLite_model/tflite_graph.pb",
    "--input_format=TENSORFLOW_GRAPHDEF",
    "--output_format=TFLITE",
    "--input_shapes=1,300,300,3",
    "--input_arrays=normalized_input_image_tensor",
    "--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3",
    "--inference_type=QUANTIZED_UINT8",
    "--std_dev_values=128",
    "--mean_values=128",
    "--change_concat_input_ranges=false",
    "--allow_custom_ops",
]

output = Popen(script, stdout=PIPE, stderr=PIPE)
stdout, stderr = output.communicate()
print(stdout)
print(stderr)


# def representative_dataset_gen():
#     ds = []
#     for imgPath in paths.list_images("./dataset_quantize"):
#         img = cv2.imread(imgPath)
#         img = cv2.resize(img, (normH, normW))
#         img = img / 128.0
#         img = img.astype(np.float32)
#         ds.append(img)
#     ds = np.array(ds)
#     print(f"[INFO] representative shape {ds.shape}")

#     images = tf.data.Dataset.from_tensor_slices(ds).batch(1)

#     for image in images.take(BATCH_SIZE):
#         yield [image]


# # Convert
# converter = tf.lite.TFLiteConverter.from_frozen_graph("TFLite_model/tflite_graph.pb")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# converter.representative_dataset = representative_dataset_gen
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = [tf.uint8]
# converter.inference_output_type = [tf.uint8]
# tflite_quant_model = converter.convert()
# print("[INFO] Successfully convert")

# # Save
# model_dir = "./TFLite_model/detect_model_quant.tflite"
# model_dir.write_bytes(tflite_quant_model)
# print("[INFO] Successfully saved")
