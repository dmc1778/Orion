import tensorflow as tf
import os
import numpy as np
try:
  data_tensor = 659 
  data = tf.identity(data_tensor)
  segment_ids_0 = 125091515651
  segment_ids_1 = False
  segment_ids_2 = 125091515651
  segment_ids = [segment_ids_0,segment_ids_1,segment_ids_2,]
  name = None
  out = tf.math.segment_mean(data=data,segment_ids=segment_ids,name=name,)
except Exception as e:
  print("Error:"+str(e))