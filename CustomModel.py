import tensorflow as tf
from constants import *
import numpy as np
from tensorflow.python.framework import tensor_shape
from CommonModel import CommonModel

class CustomModel(CommonModel):

	def __init__(self,num_classes,num_batches_per_epoch,FLAGS):
		with tf.Graph().as_default() as self.graph:
			CommonModel.__init__(self)
			self.x = tf.placeholder(tf.float32,shape=[None, IMAGE_SIZE_X, IMAGE_SIZE_Y, MODEL_INPUT_DEPTH],name='InputPlaceholder')
			self.y = tf.placeholder(tf.float32,[None, num_classes],name='GroundTruthInput')
			self._setupNetwork(num_classes,num_batches_per_epoch,FLAGS)

	def _buildNetwork(self,num_classes,FLAGS):
		weights = {'out': tf.Variable(tf.random_normal([512, num_classes],seed=opsSeed))}
		biases = {'out': tf.Variable(tf.random_normal([num_classes],seed=opsSeed))}

		conv1 = self._add_conv_layer(self.x,3,3,MODEL_INPUT_DEPTH,32,FLAGS.enable_local_response_normalization,FLAGS.use_batch_normalization,'conv1',self.is_training)
		conv2 = self._add_conv_layer(conv1,3,3,32,32,FLAGS.enable_local_response_normalization,FLAGS.use_batch_normalization,'conv2',self.is_training)
		conv2 = self._maxpool2d(conv2)

		conv3 = self._add_conv_layer(conv2,3,3,32,64,FLAGS.enable_local_response_normalization,FLAGS.use_batch_normalization,'conv3',self.is_training)
		conv4 = self._add_conv_layer(conv3,3,3,64,64,FLAGS.enable_local_response_normalization,FLAGS.use_batch_normalization,'conv4',self.is_training)
		conv4 = self._maxpool2d(conv4)

		conv5 = self._add_conv_layer(conv4,3,3,64,128,FLAGS.enable_local_response_normalization,FLAGS.use_batch_normalization,'conv5',self.is_training)
		conv6 = self._add_conv_layer(conv5,3,3,128,128,FLAGS.enable_local_response_normalization,FLAGS.use_batch_normalization,'conv6',self.is_training)
		conv6 = self._maxpool2d(conv6)

		conv6 = tf.reshape(conv6, [-1, IMAGE_SIZE_X/8 * IMAGE_SIZE_Y/8 * 128])

		fc_1 = self._add_fully_connected_layer(conv6,IMAGE_SIZE_X/8 * IMAGE_SIZE_Y/8 * 128,1024,'fc_1',self.keep_rate,self.is_training,FLAGS)
		fc_2 = self._add_fully_connected_layer(fc_1,1024,512,'fc_2',self.keep_rate,self.is_training,FLAGS)

		output = tf.matmul(fc_2, weights['out']) + biases['out']
		return output

	def _setupNetwork(self,num_classes,num_batches_per_epoch,FLAGS):
		self.prediction = self._buildNetwork(num_classes,FLAGS)
		self.add_image_standardization_ops(FLAGS)
		self.final_tensor = tf.nn.softmax(self.prediction, name=FLAGS.final_tensor_name)
		self.cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		self.create_learning_rate(FLAGS,self.global_step,num_batches_per_epoch)
		optimizer = self.create_optimizer(FLAGS,self.learning_rate)

		self.train_step_op = optimizer.minimize(self.cross_entropy_mean,global_step=self.global_step)
		self.add_evaluation_step()