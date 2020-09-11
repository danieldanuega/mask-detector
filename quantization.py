import numpy as np
import tensorflow as tf
from imutils import paths
import cv2

"""
This is the quantization script for tflite model float32 to int8 model.
I use the original dataset by Yifan Wang for calibration purpose https://drive.google.com/drive/folders/1w-zoOuSauw5xBhOrfZpjcglvDH14mlGu.

The dataset calibration is not included in this repo, please see the issue pages of Yifan Wang for further information. 
https://github.com/wangyifan411/Face-Mask-Type-Detector/issues/1

FYI:
the model that is used by Yifan Wang is SSD_Mobilenet_V2_COCO, where the size is 300x300.
You could find more information about size, batch_size, etc, of the pre-trained model in tensorflow object detection model zoo 1.
This repo use Tensorflow 1
"""

SIZE = (300, 300)
BATCH_SIZE = 24


def representative_dataset_gen():
    ds = []
    for imgPath in paths.list_images("./dataset_quantize"):
        img = cv2.imread(imgPath)
        img = cv2.resize(img, SIZE)
        img = img / 255.0
        img = img.astype(np.float32)
        ds.append(img)
    ds = np.array(ds)

    images = tf.data.Dataset.from_tensor_slices(ds).batch(1)

    for image in images.take(BATCH_SIZE):
        yield [image]


# Convert
converter = tf.lite.TFLiteConverter.from_saved_model(
    "./output_inference_graph_mobile.pb/saved_model"
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

# Save
model_dir = "./TFLite_model/detect_model_quant.tflite"
model_dir.write_bytes(tflite_quant_model)
