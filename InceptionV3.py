import tensorflow as tf
from constants import *
import numpy as np
from tensorflow.python.framework import tensor_shape
from CommonModel import CommonModel

class InceptionV3(CommonModel):

	bottleneckTensor = None
	bottleneckInput = None
	jpeg_data_tensor = None

	def __init__(self,modelPath):
		with tf.Graph().as_default() as self.graph:
			CommonModel.__init__(self)
			self._create_inception_graph(modelPath)

	def _create_inception_graph(self,modelPath):
		with tf.gfile.FastGFile(modelPath, 'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				self.bottleneckTensor, self.jpeg_data_tensor, resized_input_tensor, self.decoded_jpeg_data_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,RESIZED_INPUT_TENSOR_NAME,DECODED_JPEG_DATA_TENSOR_NAME]))

	def add_final_training_ops(self,class_count, final_tensor_name, optimizer_name, num_batches_per_epoch, FLAGS):
		with self.graph.as_default():
			with tf.name_scope('input'):
				self.x = tf.placeholder_with_default(self.bottleneckTensor, shape=[None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
				self.y = tf.placeholder(tf.float32,[None, class_count],name='GroundTruthInput')

			#layer_name = 'final_minus_2_training_ops'
			#logits_final_minus_2 = self._add_fully_connected_layer(self.bottleneckInput,BOTTLENECK_TENSOR_SIZE,FINAL_MINUS_2_LAYER_SIZE,layer_name,self.keep_rate,self.is_training,FLAGS)

			#layer_name = 'final_minus_1_training_ops'
			#logits_final_minus_1 = self._add_fully_connected_layer(logits_final_minus_2,FINAL_MINUS_2_LAYER_SIZE,FINAL_MINUS_1_LAYER_SIZE,layer_name,self.keep_rate,self.is_training,FLAGS)

			layer_name = 'final_minus_1_training_ops'
			logits_final_minus_1 = self._add_fully_connected_layer(self.x,BOTTLENECK_TENSOR_SIZE,FINAL_MINUS_1_LAYER_SIZE,layer_name,self.keep_rate,self.is_training,FLAGS)

			layer_name = 'final_training_ops'
			with tf.name_scope(layer_name):
			    with tf.name_scope('weights'):
			    	initial_value = tf.truncated_normal([FINAL_MINUS_1_LAYER_SIZE, class_count],stddev=0.001)
			      	layer_weights = tf.Variable(initial_value, name='final_weights')
			    with tf.name_scope('biases'):
			      	layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
			    with tf.name_scope('Wx_plus_b'):
					logits = tf.matmul(logits_final_minus_1, layer_weights) + layer_biases

			self.final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

			with tf.name_scope('cross_entropy'):
				self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
			    	with tf.name_scope('total'):
			    		self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy)

			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.create_learning_rate(FLAGS,self.global_step,num_batches_per_epoch)

			with tf.name_scope('train'):
				optimizer = self.create_optimizer(FLAGS,self.learning_rate)
				self.train_step_op = optimizer.minimize(self.cross_entropy_mean,global_step=self.global_step)


	def run_bottleneck_on_image(self,sess, image_data):
	  #bottleneck_values = sess.run(self.bottleneckTensor,{self.jpeg_data_tensor: image_data})
	  bottleneck_values = sess.run(self.bottleneckTensor,{self.decoded_jpeg_data_tensor: image_data})
	  bottleneck_values = np.squeeze(bottleneck_values)
	  return bottleneck_values
