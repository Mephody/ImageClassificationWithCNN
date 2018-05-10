# ImageClassificationWithCNN
Simple CNN (Convolutional Neural Network) image classififcation using Keras and TensorFlow.

Report.zip contains report .gif animations demonstrating the work of neural network.
ReportRU.pdf contains report about NN development and testing in russian language.

NeuralNetwork.py - CNN training script.
OpenCV_camera.py - CNN testing script.

mnist_model.h5 and mnist_model.json - already trained CNN.

Marcel_Test_And_Train - dataset for training and testing CNN.

Some words about CNN.
It's easy to read code for CNN architecture understanding, but nevertheless i'd want to describe its main structure.

Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer) -> Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer) -> Conv [128] -> Conv [128] -> Pool (with dropout on the pooling layer) -> flatten to 1D, apply FC -> ReLU (with dropout) -> softmax

Some words about unpacking.

You need to unpack Marcel_Test_And_Train.zip archive in your working directory (the same directory where python script lays).

