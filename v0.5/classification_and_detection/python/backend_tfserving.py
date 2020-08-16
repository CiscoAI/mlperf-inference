"""
tensorflow serving backend (https://github.com/tensorflow/tensorflow)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import os

import grpc
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import backend


class BackendTFServing(backend.Backend):
    def __init__(self):
        super(BackendTFServing, self).__init__()

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tfserving"

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        # there is no input/output meta data i the graph so it need to come from config.
        if not inputs:
            raise ValueError("BackendTensorflowServing needs inputs")
        if not outputs:
            raise ValueError("BackendTensorflowServing needs outputs")
        self.outputs = outputs
        self.inputs = inputs

        self.tfserving_endpoint = os.getenv("TFSERVING_ENDPOINT")
        if not self.tfserving_endpoint:
            raise ValueError("Expected environment variable: TFSERVING_ENDPOINT")
        self.tfserving_model_name = os.getenv("TFSERVING_MODEL_NAME")
        if not self.tfserving_model_name:
            raise ValueError("Expected environment variable: TFSERVING_MODEL_NAME")
        self.tfserving_model_signature = os.getenv("TFSERVING_MODEL_SIGNATURE")
        if not self.tfserving_model_signature:
            raise ValueError("Expected environment variable: TFSERVING_MODEL_SIGNATURE")

        channel = grpc.insecure_channel(self.tfserving_endpoint)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        return self

    def predict(self, feed):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.tfserving_model_name
        request.model_spec.signature_name = self.tfserving_model_signature
        for input_name in self.inputs:
            data = feed[input_name]
            request.inputs[input_name].CopyFrom(
                tf.make_tensor_proto(data, shape=data.shape))
        response = self.stub.Predict(request, 10.0)
        result = []
        for output_name in self.outputs:
            result.append(tf.make_ndarray(response.outputs[output_name]))
        return result
