# ece447-Srivastava-Paper
**ECE 447 Paper Project**

Daksh - Theory lead
Fawaz - Results lead
Nathan - Experiment lead
Kulnoor - Presentation lead

# Dropout Reproduction

This repository contains a simple PyTorch script that demonstrates the effect of Dropout regularization on a neural network. 

To clearly show the benefits of Dropout, the script intentionally forces a model to overfit by training it on a small subset (5,000 images) of the Fashion-MNIST dataset. It then trains two identical Multilayer Perceptrons (MLPs), one with Dropout and one without, and plots their training and testing accuracies side-by-side.

## Prerequisites

I was using python 3.13 
You will need the following libraries:
* `torch`
* `torchvision`
* `matplotlib`

You can install the required dependencies using pip:

* `pip install torch torchvision matplotlib`

## How to Run

* `python3 experiment.py` (When in correct directory)
* If you wish to change # of epochs, learning rate, or dropout rate, the values are contained in lines 10-12, but the recommended values are 50, 0.01, and 0.5 respectively

## Expected Outputs

1. The code will automatically download the Fashion-MNIST dataset into a local ./data folder
2. It will print train accuracy and test accuracy side by side in the terminal for each epoch
3. Once training finishes, it will make a plot window that shows a graph of the two accuracies. There should be a clear difference between test and train acc.