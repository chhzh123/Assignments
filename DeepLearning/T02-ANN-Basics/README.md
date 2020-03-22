# Week 02 - Artificial Neural Networks: Basics

## Neuron
* Input -> Hidden -> Output
* Weights
* Bias

## FCN
* Stack the layers
* Activation funcion
	* e.g.
		* Sigmoid
		* Tanh
		* ReLU
	* Linear -> Nonlinear
	* Complex relations
* Issues
	* Great # of params
	* Not feasible

## CNN
* Convolution
	* f: image
	* g: kernel/filter
	* (f * g): feature map
	* channels
* Higher-level filter
	* stride: get larger
	* pooling: max/avg
* Stack all (end-to-end learning)
	* low -> mid -> high-level feat.
* Researchers
	* Yoshua Bengio
		* Prob models, GAN
	* Geoffrey Hinton
		* BP, CNN, Boltzmann
	* Yann LeCun
		* CNN, BP
	* Jürgen Schmidhuber
		* LSTM
* Models
	* LeNet [1989]
	* AlexNet [2012]
	* VGG [2014]
	* GoogleNet [2014]
	* ResNet [2015]