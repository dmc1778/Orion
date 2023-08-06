results = dict()
import tensorflow as tf
import numpy as np
from tensorflow.python.data.experimental.ops import shuffle_ops
try:
  try:
    with tf.device('/CPU'):
      buffer_size = 0
      out = shuffle_ops.shuffle_and_repeat(buffer_size=buffer_size,)
  except Exception as e:
    print("Error:"+str(e))
  try:
    with tf.device('/GPU:0'):
      shuffle_ops.shuffle_and_repeat(buffer_size=buffer_size,)
  except Exception as e:
    print("Error:"+str(e))
except Exception as e:
  print("Error:"+str(e))

print(results)