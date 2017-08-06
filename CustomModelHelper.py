import os.path
from constants import *
from CustomModel import *

def create_custom_graph(num_classes,num_batches_per_epoch,FLAGS):
    custom_model = CustomModel(num_classes,num_batches_per_epoch,FLAGS)
    return custom_model
