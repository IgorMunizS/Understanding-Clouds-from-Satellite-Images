# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""ResNet models for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import resnet

from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.applications.resnet50.ResNet50',
              'keras.applications.resnet.ResNet50',
              'keras.applications.ResNet50')
@keras_modules_injection
def ResNet50(*args, **kwargs):
  return resnet.ResNet50(*args, **kwargs)


@keras_export('keras.applications.resnet.ResNet101',
              'keras.applications.ResNet101')
@keras_modules_injection
def ResNet101(*args, **kwargs):
  return resnet.ResNet101(*args, **kwargs)


@keras_export('keras.applications.resnet.ResNet152',
              'keras.applications.ResNet152')
@keras_modules_injection
def ResNet152(*args, **kwargs):
  return resnet.ResNet152(*args, **kwargs)


@keras_export('keras.applications.resnet50.decode_predictions',
              'keras.applications.resnet.decode_predictions')
@keras_modules_injection
def decode_predictions(*args, **kwargs):
  return resnet.decode_predictions(*args, **kwargs)


@keras_export('keras.applications.resnet50.preprocess_input',
              'keras.applications.resnet.preprocess_input')
@keras_modules_injection
def preprocess_input(*args, **kwargs):
  return resnet.preprocess_input(*args, **kwargs)