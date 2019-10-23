close all;clear all;clc;
% I = imread('fig/Lenna.png');
I = imread('fig/Fig0635.tif');
[m,n,k] = size(I);

% all channels
newI = histeq(I);
    
% different channels
newI2 = zeros(m,n,3);
newI2(:,:,1) = histeq(I(:,:,1));
newI2(:,:,2) = histeq(I(:,:,2));
newI2(:,:,3) = histeq(I(:,:,3));

figure,
subplot(131),imshow(uint8(I));
title('原图')
subplot(132),imshow(uint8(newI));
title('均衡化后（全通道）')
subplot(133),imshow(uint8(newI2));
title('均衡化后（分通道）')

function g = histeq(img)
	A = zeros(1,256);
	for i = 1:256
		A(i) = sum(img == (i-1),'all');
	end
	A = double(A);
	A = A / prod(size(img));
	cumulation = zeros(1,256);
	for i = 2:256
	    cumulation(i) = cumulation(i-1) + A(i);
    end
    g = uint8(cumulation(img+1)*255);
end