# Week 05 - Segmentation

## Sematic Segmentation
* classifying each pixel
* applications
	* self-driving
	* smart home
	* medical analysis
* traditional ideas
	* surrounding patch
	* \# Class channels

## Fully Convolutional Network
* **Encoder-decoder**
* Transposed convolution (deconv)
	* learnable upsampling
	* stride, padding
* Long [CVPR'15]
	![fcn8](fig/fcn8.jpg)
	* CrossEntropy per pixel

## U-net [MICCAI'15]
* More encoding layers
![unet](fig/unet.jpg)
* More kernels in upsampling
* Encoding => decoding (connect)
* Upsampled + Encoded features
* Problems
	* × global info
* Extensions
	* Tiramisu net = U-Net + Dense ResNet [CVPR'17]
	* PSPNet [CVPR'17]
		![pspnet](fig/pspnet.jpg)
		* dilated conv
		* pyramid pooling
		* upsampled + encoded

## DeepLab [arXiv'17]
* v1
	![deeplabv1](fig/deeplabv1.jpg)
	* Conv + pooling
	* Dilated conv
	* bilinear interpolation -> resize
	* Post-processing: fully connected conditional random field
* v2
	* fuse paths of dilated conv w/ diff rates
	* multi-scale inputs
	* pre-training
* v3
	* global pooling
	* batch norm
	* bootstrapping
* v3+
	* v3 + decoder
* Extension
	* Gated shape CNN [ICCV'19]
	* Auto-DeepLab [arXiv'19]

## Future directions
* Prior knowledge
* Small annotated data
* Transfer learning
* Weakly supervised
* Combinations