# Week 04 - CNN Training Issues

## Training
* Pre-set hyper-params
* Initialize model params
* Repeat iterations
	* Shuffle whole training data
	* For each mini-batch
		* Load data
		* Compute gradient
		* Update params
* Save model

## Issues
* Gradient exploding
	* |g'(u_i)|<=1
		* sigmoid
		* tanh
		* ReLU
	* weight initialization: |w_i|<=1
		* Xavier
		* Kaiming
	* weight re-normalization
	* rescale x s.t. |x|<=1
* Gradient vanishing
	* ReLU
	* weight init
		* normal(0,sigma^2)
		* uniform(-a,a)
	* weight re-norm
* Diff dist of mini-batch
	* Batch normalization (BN)
	* Group normalization (GN)

## Deeper networks
* ResNet [CVPR'16]
	* Residual mapping
* DenseNet [CVPR'17]
* Wide ResNet [BMVC'16]
* ResNeXt [CVPR'17]
* SENet [CVPR'18]
* PNASNet [ECCV'18]