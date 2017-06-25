# InceptionV3-Tensorflow-Retraining

In this repository, we train a Deep Convolutional Neural Net for participating in the [Kaggle CIFAR-10] (https://www.kaggle.com/c/cifar-10). While the challenge had ended years ago, it's a classic dataset to try out :)

We trained the model on the <b>CIFAR-10 training dataset</b> and tested it on the Kaggle CIFAR-10 Test Set by submitting it on the challenge page. 

<b>Accuracy</b>
This model achieved an accuracy of <b>91.4%</b> over the test set.

<b>Code</b>
This model is inspired by the [Image Retraining](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/image_retraining) example. Some code has been trimmed off and refactored as per our use case.

In this code, we added untrained (depending on the number of classes) fully connected layers to the loaded InceptionV3 graph. The loaded InceptionV3 model is NOT FINE TUNED. It is only used to generate the features for the images. However, the fully connected layers are trained.

Run the following script to train the network.
```python
python trainer.py
```

Run the following script for testing the network. This script classifies the images from test dataset and dumps the results in a file named "kaggle_result.txt" as per the format expected by Kaggle. Just upload the file on the submission page to get it evaluated. Please provide the path to the trained model and place all test images in a folder named "test".

```python
python tester.py
```

<b> Dataset </b>

We downloaded the dataset from the Kaggle [CIFAR-10 classification challenge](https://www.kaggle.com/c/cifar-10/data). The CIFAR-10 training dataset contains 50,000 training images (5000 images per class).

The Kaggle CIFAR-10 Test Set contains 3,00,000 images out of which 10,000 are used for evaluation. The remaining 2,90,000 have been added to prevent cheating :) 

For details of the dataset and challenge, please [visit this](https://www.kaggle.com/c/cifar-10)

The code expects the data to be organized in a parent folder with images of different classes in different folders. The CIFAR training dataset is not in this format, so we wrote a small script "images_organizer.py" to segregate the dataset into different folder, each belonging to a class. 

For Example,</br>
cifar10Dataset - Parent Directory </br>
cifar10Dataset/airplane </br>
cifar10Dataset/dog </br>
