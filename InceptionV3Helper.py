import os.path
from constants import *
from InceptionV3 import *

def create_inception_graph(num_classes,num_batches_per_epoch,FLAGS):
    modelFilePath = os.path.join(FLAGS.imagenet_inception_model_dir, INCEPTION_MODEL_GRAPH_DEF_FILE)
    inceptionV3 = InceptionV3(modelFilePath)
    inceptionV3.add_final_training_ops(num_classes,FLAGS.final_tensor_name,FLAGS.optimizer_name,num_batches_per_epoch, FLAGS)
    inceptionV3.add_evaluation_step()
    return inceptionV3

def setup_image_distortion_ops(inceptionV3,FLAGS):
    inceptionV3.add_input_distortions(FLAGS.flip_left_right, FLAGS.random_crop,FLAGS.random_scale, FLAGS.random_brightness)
