close all;clear all;clc;

% PROJECT 04-02
% (a)
I = imread('Fig0418(a).tif');
I = double(I);
[m,n] = size(I);
fftI = fft2(centerize(I));
sp = spectrum(fftI);

% (c)
s = sum(sum(abs(fftI)));
avg = s / (m * n);

% (b)
figure,
subplot(121),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(122),imshow(sp.^0.2,[]);
title('Fig.4.18(a)傅里叶谱')

% PROJECT 04-03 (b)
gimg = gauss_lowpass(I,m/2,n/2,15);
figure,
subplot(121),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(122),imshow(uint8(gimg));
title('Fig.4.18(a)高斯低通滤波后')

% PROJECT 04-04
% (a)
simg = I - gimg;
figure,
subplot(121),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(122),imshow(uint8(simg));
title('Fig.4.18(a)钝化模板')

% (b)
simg2 = I - gauss_lowpass(I,m/2,n/2,100);
figure,
subplot(121),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(122),imshow(uint8(simg2));
title('Fig.4.18(a)钝化模板 (sigma=100)')

% PROJECT 04-05
I1 = imread('Fig0441(a).jpg');
I2 = imread('Fig0441(b).jpg');
[m1,n1] = size(I1);
[m2,n2] = size(I2);
P = 298;
Q = 298;
img1 = zeros(P,Q);
img2 = zeros(P,Q);
img1(1:m1,1:n1) = I1(1:m1,1:n1);
img2(1:m2,1:n2) = I2(1:m2,1:n2);
cimg1 = centerize(img1);
cimg2 = centerize(img2);
f1 = fft2(cimg1);
f2 = fft2(cimg2);
rel = conj(f1).* f2;
% rel = f2 .* conj(f1);
newI = recover(ifft2(rel));
figure,
subplot(131),imshow(uint8(I1));
title('Fig.4.41(a)原图')
subplot(132),imshow(uint8(I2));
title('Fig.4.41(b)原图')
subplot(133),imshow(uint8(newI.^0.3));
title('Fig.4.41图像相关')

% PROJECT 04-03 (a)
function g = gauss_lowpass(img,center_x,center_y,sig)
	[M,N] = size(img);
	[X,Y] = meshgrid(1:M,1:N);
	D = (X - center_x).^2 + (Y - center_y).^2;
	H = exp(-D/(2*sig^2));
	cimg = centerize(img);
	f = fft2(cimg);
	g = centerize(real(ifft2(H.*f)));
end

% PROJECT 04-01
% (a)
function g = centerize(img)
	[M,N] = size(img);
	[X,Y] = meshgrid(1:M,1:N);
	ones = (-1).^(X+Y);
	g = ones.*img;
end

% (b)
function g = mul_real(A,c)
	% g = c * real(A) + c * imag(A) * i;
	g = c * A;
end

% (c)
function g = inverse_fft(A)
	g = ifft2(A);
end

% (d)
function g = recover(A)
	g = centerize(real(A));
end

% (e)
function g = spectrum(A)
	g = abs(A);
end