import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

startTime = datetime.now()

shape = (int(100000), int(100000))
with tf.device("/gpu"):
    random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

result = sum_operation
print(result)

print("\n" * 2)
print("Time taken:", datetime.now() - startTime)
print("\n" * 2)