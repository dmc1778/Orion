import tensorflow as tf
import os
import numpy as np
try:
  input_tensor = tf.constant(-78106177161849, shape=[4, 4], dtype=tf.float32,)
  input = tf.identity(input_tensor)
  diagonal_0_0 = 2.0
  diagonal_0_1 = 3.0
  diagonal_0_2 = 4.0
  diagonal_0_3 = 5.0
  diagonal_0 = [diagonal_0_0,diagonal_0_1,diagonal_0_2,diagonal_0_3,]
  diagonal_1_0 = -1.0
  diagonal_1_1 = -2.0
  diagonal_1_2 = -3.0
  diagonal_1_3 = -4.0
  diagonal_1 = [diagonal_1_0,diagonal_1_1,diagonal_1_2,diagonal_1_3,]
  diagonal = [diagonal_0,diagonal_1,]
  k_0 = None
  k_1 = None
  k = [k_0,k_1,]
  align = "`o0Wap"U[9"
  name = "set_diag"
  out = tf.raw_ops.MatrixSetDiagV3(input=input,diagonal=diagonal,k=k,align=align,name=name,)
except Exception as e:
  print("Error:"+str(e))