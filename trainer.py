import argparse
import preprocessor as preprocessor
from constants import *

import os.path
from datetime import datetime
from DatasetBatcher import *
import DatasetManager as DatasetManager
from tensorflow.python.framework import graph_util
import time
from osUtils import *
from InceptionV3Helper import *
from CustomModelHelper import *


def evaluate_accuracy(sess,phase,model,datasetBatcher,FLAGS,epoch_index):
    batch_size = 0
    if phase == "training":
        datasetBatcher.reset_training_offset()
        batch_size = FLAGS.train_batch_size
    elif phase == "testing":
        datasetBatcher.reset_testing_offset()
        batch_size = FLAGS.test_batch_size
    elif phase == "validation":
        datasetBatcher.reset_validation_offset()
        batch_size = FLAGS.validation_batch_size

    image_paths,ground_truth,labels = datasetBatcher.get_next_batch(phase,batch_size)
    batch_index = 0
    accuracy = 0
    cross_entropy_value = 0
    num_samples = 0
    while image_paths is not None:
        batch_index = batch_index + 1
        num_samples_in_batch = 0
        if FLAGS.model_type == "inception":
            bottlenecks = DatasetManager.get_random_cached_bottlenecks_new(sess,image_paths,labels,FLAGS.bottleneck_dir,model)
            num_samples_in_batch = len(bottlenecks)
            accuracy_batch, cross_entropy_value_batch = model.evaluate(sess,bottlenecks,ground_truth)
        else:
            images_data = DatasetManager.get_image_data(image_paths,model,sess)
            num_samples_in_batch = images_data.shape[0]
            accuracy_batch, cross_entropy_value_batch = model.evaluate(sess,images_data,ground_truth)

        num_samples = num_samples + num_samples_in_batch
        accuracy = accuracy + accuracy_batch
        cross_entropy_value = cross_entropy_value + cross_entropy_value_batch
        image_paths,ground_truth,labels = datasetBatcher.get_next_batch(phase,batch_size)

    if batch_index == 0:
        print "No samples to evaluate in this phase:" + phase
    else:
        accuracy = accuracy * 100/batch_index
        print('%s: Step %d: %s Accuracy = %.1f%%' % (datetime.now(), epoch_index,phase,accuracy))
        print('%s: Step %d: %s Cross entropy = %f' % (datetime.now(), epoch_index,phase,cross_entropy_value/batch_index))
    return accuracy,cross_entropy_value


def train_an_epoch(sess,model,datasetBatcher,FLAGS):
    datasetBatcher.reset_training_offset(FLAGS.shuffle_dataset_every_epoch)
    train_image_paths,train_ground_truth,train_labels = datasetBatcher.get_next_training_batch(FLAGS.train_batch_size)
    while train_image_paths is not None:
        if FLAGS.model_type == "inception":
            if DatasetManager.should_distort_images(FLAGS):
                train_bottlenecks = DatasetManager.get_random_distorted_bottlenecks(sess,train_image_paths,model)
            else:
                train_bottlenecks = DatasetManager.get_random_cached_bottlenecks_new(sess,train_image_paths,train_labels,FLAGS.bottleneck_dir,model)
            model.train_step(sess,train_bottlenecks,train_ground_truth,FLAGS.dropout_keep_rate)
        else:
            train_images_data = DatasetManager.get_image_data(train_image_paths,model,sess)
            model.train_step(sess,train_images_data,train_ground_truth,FLAGS.dropout_keep_rate)

        train_image_paths,train_ground_truth,train_labels = datasetBatcher.get_next_training_batch(FLAGS.train_batch_size)
    datasetBatcher.reset_training_offset()

def train_graph(model,datasetBatcher,FLAGS):
    with tf.Session(graph=model.get_graph()) as sess:
        start_time = str(int(time.time()))
        init = tf.global_variables_initializer()
        sess.run(init)
        print "Training the model with dropout rate:" + str(FLAGS.dropout_keep_rate)
        for i in range(FLAGS.how_many_training_steps):
            print "Epoch..."+str(i)+"/"+str(FLAGS.how_many_training_steps)
            train_an_epoch(sess,model,datasetBatcher,FLAGS)
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy,_ = evaluate_accuracy(sess,"training",model,datasetBatcher,FLAGS,i)
                validation_accuracy,_ = evaluate_accuracy(sess,"validation",model,datasetBatcher,FLAGS,i)
                model_name = "model_"+str(train_accuracy)+"_"+str(validation_accuracy)+".pb"
                save_graph(sess,model.get_graph(),start_time,model_name,FLAGS)
        evaluate_accuracy(sess,"testing",model,datasetBatcher,FLAGS,i)
        save_graph(sess,model.get_graph(),start_time,"model_final.pb",FLAGS)

def save_graph(sess,graph,prefix,model_name,FLAGS):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    sub_dir_path = os.path.join(FLAGS.output_graph, prefix)
    ensure_dir_exists(sub_dir_path)
    output_graph_path = os.path.join(sub_dir_path,model_name)
    print "Saving the graph at:"+ output_graph_path
    with tf.gfile.FastGFile(output_graph_path, 'wb') as f:
      f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type',type=str,default='inception',help="custom,inception")
  parser.add_argument('--output_graph',type=str,default='./tmp/output_graph', help='Where to save the trained graph.')
  parser.add_argument('--output_labels',type=str,default='./tmp/output_labels.txt',help='Where to save the trained graph\'s labels.')
  parser.add_argument('--summaries_dir',type=str,default='./tmp/retrain_logs',help='Where to save summary logs for TensorBoard.')
  parser.add_argument('--how_many_training_steps',type=int,default=500,help='How many training steps to run before ending.')
  parser.add_argument('--imagenet_inception_model_dir',type=str,default='./imagenetInception',help="""Path to classify_image_graph_def.pb,imagenet_synset_to_human_label_map.txt, and imagenet_2012_challenge_label_map_proto.pbtxt.""")
  parser.add_argument('--bottleneck_dir',type=str,default='./tmp/bottleneck',help='Path to cache bottleneck layer values as files.')
  parser.add_argument('--final_tensor_name',type=str,default='final_result',help="""The name of the output classification layer in the retrained graph.""")
  parser.add_argument('--image_dir',type=str,default='cifar10DatasetSample',help='Path to folders of labeled images.')
  parser.add_argument('--eval_step_interval',type=int,default=5,help='How often to evaluate the training results.')
  parser.add_argument('--print_misclassified_test_images',default=False, help="Whether to print out a list of all misclassified test images.",action='store_true')

  #Learning Rate and Optimizers
  parser.add_argument('--optimizer_name',type=str,default="sgd",help='Optimizer to be used: sgd,adam,rmsprop')
  parser.add_argument('--learning_rate_decay_factor',type=float,default=0.16,help='Learning rate decay factor.')
  parser.add_argument('--learning_rate',type=float,default=0.1,help='Initial learning rate.')
  parser.add_argument('--rmsprop_decay',type=float,default=0.9,help='Decay term for RMSProp.')
  parser.add_argument('--rmsprop_momentum',type=float,default=0.9,help='Momentum in RMSProp.')
  parser.add_argument('--rmsprop_epsilon',type=float,default=1.0,help='Epsilon term for RMSProp.')
  parser.add_argument('--num_epochs_per_decay',type=int,default=30,help='Epochs after which learning rate decays.')
  parser.add_argument('--learning_rate_type',type=str,default="const",help='exp_decay,const')

  #Normalizations/Regularizations
  parser.add_argument('--use_batch_normalization',type=bool,default=False,help='Control the use of batch normalization')
  parser.add_argument('--dropout_keep_rate',type=float,default=0.5)
  parser.add_argument('--enable_local_response_normalization',type=bool,default=False)

  #Batch Sizes
  parser.add_argument('--train_batch_size',type=int,default=64,help='How many images to train on at a time.')
  parser.add_argument('--test_batch_size',type=int,default=32)
  parser.add_argument('--validation_batch_size',type=int,default=32)

  parser.add_argument('--testing_percentage',type=int,default=0,help='What percentage of images to use as a test set.')
  parser.add_argument('--validation_percentage',type=int,default=20,help='What percentage of images to use as a validation set.')

  #distortions
  parser.add_argument('--apply_distortions',default=False,help="Apply distortions to images while training.")
  parser.add_argument('--shuffle_dataset_every_epoch',default=True,help="Shuffle the training dataset at every epoch.")
  parser.add_argument('--flip_left_right',default=True,help="Whether to randomly flip half of the training images horizontally.",action='store_true')
  parser.add_argument('--random_crop',type=int,default=20,help="A percentage determining how much of a margin to randomly crop off the training images.")
  parser.add_argument('--random_scale',type=int,default=20,help="A percentage determining how much to randomly scale up the size of the training images by.")
  parser.add_argument('--random_brightness',type=int,default=20,help="A percentage determining how much to randomly multiply the training image input pixels up or down by.")


  FLAGS, unparsed = parser.parse_known_args()
  preprocessor.setup(FLAGS)

  image_map = DatasetManager.readDataset(FLAGS)
  num_classes = len(image_map.keys())

  datasetBatcher = DatasetBatcher(image_map,FLAGS.image_dir)
  num_training_batches = datasetBatcher.number_of_training_batches(FLAGS.train_batch_size)

  model = None
  if FLAGS.model_type == "inception":
      print "Creating Inception Graph..."
      model = create_inception_graph(num_classes,num_training_batches, FLAGS)
      if DatasetManager.should_distort_images(FLAGS):
          setup_image_distortion_ops(model,FLAGS)
  else:
      print "Creating Custom Graph..."
      model = create_custom_graph(num_classes,num_training_batches, FLAGS)

  train_graph(model,datasetBatcher,FLAGS)
