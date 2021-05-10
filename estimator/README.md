# Deep learning based on TensorFlow's estimator framework

TensorFlow's estimator module is a powerful way to create deep learning programs.  This repository abstracts this functionality further to provie a software framework for training, evaluating and testing deep learning methods, specifically with a focus on medical imaging.

Rather than creating many separate models that repeat functionality, this framework attempts to provide an interface for rapid parameterization of models along with a simple common interface for training and testing these models. Many different models can be created through the command line interface. And, new models, loss functions, and optimizers may be added easily as users require custom functionality.

## Structure of the code

This framework takes an object-oriented software design approach. The framework abstracts the machine learning paradigm into the following specific modules:
* **Estimator:** the driver program for the specific type of machine learning task, e.g. classification or regression.
* **Model:** the specific deep learning network architecture used by the estimator, e.g. the U-Net convolutional neural network.
* **Loss function:** the function to guide network learning, e.g. cross entropy or L2 norm.
* **Optimizer:** the optimization method to minimize the specified loss function, e.g. Gradient Descent.
* **Data:** method for data IO and data fetching, used for training, evaluation, and testing data.

## Data processing

The framework takes a patch-based approach to data handling. TensorFlow can handle N-D data as tensors, but during training this data must first be processed to have common sizes, which is often not the case with heterogenous medical image data. Therefore, we provide functionailty to load the raw image data (currently only NIfTI, .jpg, .png, and .fits file formats are supported) and then sample patches of the same size from this data, if desired.

## Running the code

See the example directory for a tutorial how to run an example network using real image data.

## Software dependencies

Requires nibabel, PILImage, astropy


## TODO

- [ ] Add sample script using the sample data
- [ ] Proper documentation of the code



