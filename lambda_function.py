import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
import numpy as np
import json

preprocessor = create_preprocessor("resnet50", target_size=(200, 200))
interpreter = tflite.Interpreter(model_path="hair-classifier.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

classes = ("straight", "wavy", "curly", "dreadlocks", "kinky")

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def predict(url):
    x = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_index)[0].tolist()
    probabilities = softmax(logits)
    dict_predictions = dict(zip(classes, probabilities.tolist()))

    return dict_predictions

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return json.dumps(result)