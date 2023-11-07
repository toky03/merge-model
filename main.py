import time


import tensorflow as tf
import tensorflow_hub as hub

def save_tf_lite_model(layered_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(layered_model)
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS] # enable TensorFlow ops.
    tflite_model = converter.convert()
    with open('combined_model.tflite', 'wb') as f:
        f.write(tflite_model)


class ObjectDetectionLayer(tf.keras.layers.Layer):
    def __init__(self, detection_model, **kwargs):
        super(ObjectDetectionLayer, self).__init__(**kwargs)
        self.detection_model = detection_model
    
    def call(self, inputs):
        return self.detection_model(inputs)
    

class ConditionalClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, classification_model, **kwargs):
        super(ConditionalClassificationLayer, self).__init__(**kwargs)
        self.classification_model = classification_model
    
    def call(self, inputs):
         # detect only cats
        category_to_classify = 14
        input_image, detection_boxes, detection_classes = inputs

        category_indices = tf.where(tf.equal(detection_classes, category_to_classify))

        # https://www.tensorflow.org/api_docs/python/tf/gather_nd
        filtered_boxes = tf.gather_nd(detection_boxes, category_indices)

        def perform_classification():
            cropped_regions = tf.image.crop_and_resize(input_image, filtered_boxes, box_indices=tf.range(tf.shape(filtered_boxes)[0]), crop_size=(160, 160))
            classification_output = self.classification_model(cropped_regions)
            return classification_output
        
        classification_output = tf.cond(tf.shape(category_indices)[0] > 0, perform_classification, lambda: tf.constant([], dtype=tf.float32))

        return classification_output



def main():
    classification_model = tf.keras.models.load_model("layered_model")
    # object_detection_model = tf.keras.models.load_model("../object_detection_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model")

    object_detection_model_url = "https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1"
    object_detection_model = hub.load(object_detection_model_url)

    # TODO the Issue is with the shape(None, None, 3) which should be fixed with 640, 640, 3 as coral does not allow dynamic input sizes
    # Error Message: ERROR: Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors.
    input_image = tf.keras.layers.Input(shape=(None, None, 3), dtype=tf.uint8, name='image_input')
    object_detection_layer = ObjectDetectionLayer(object_detection_model)
    detection_results = object_detection_layer(input_image)
    detection_boxes = detection_results['detection_boxes']
    detection_classes = detection_results['detection_classes']

    conditional_classification_layer = ConditionalClassificationLayer(classification_model)

    classification_output = conditional_classification_layer([input_image, detection_boxes, detection_classes])

    combined_model = tf.keras.models.Model(inputs=input_image, outputs=classification_output)

    combined_model.compile()
    combined_model.summary()

    combined_model.save('combined_model')

    save_tf_lite_model(combined_model)





if __name__ == '__main__':
    main()

