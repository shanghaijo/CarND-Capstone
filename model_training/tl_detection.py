import os
import sys
import tensorflow as tf

# change dir and path as specified in installation instructions
os.chdir(os.path.join(os.path.dirname(tf.__file__), "models", "research"))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "slim"))

from object_detection.utils import dataset_util

