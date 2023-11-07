# Code to merge a Object Detection Model with a classification model

## The challange is to perform the classification only on a certain condition

# Build model
`python3 main.py`

# Convert to edgetpu compatible model with [edgetpu_compiler](https://coral.ai/docs/edgetpu/compiler/#usage)
`edgetpu_compiler combined_model.tflite`