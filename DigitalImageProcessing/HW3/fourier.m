close all;clear all;clc;

% PROJECT 04-02
% (a)
I = imread('Fig0418(a).tif');
I = double(I);
[m,n] = size(I);
fftI = fft2(centerize(I));
sp = spectrum(fftI);

% (b)
figure,
subplot(121),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(122),imshow(uint8(sp.^0.4),[]); % (log(1 + sp),[]);
title('Fig.4.18(a)傅里叶谱')

% (c)
s = sum(sum(I));
avg = s / (m * n)
fftOrig = fft2(I);
avg_fourier = fftOrig(1,1) / (m*n)

% PROJECT 04-03 (b)
gimg = gauss(I,15,1);
figure,
subplot(121),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(122),imshow(uint8(gimg));
title('Fig.4.18(a)高斯低通滤波后')

% PROJECT 04-04
% (a)
simg = I - gimg;
simg1 = gauss(I,15,0);
figure,
subplot(131),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(132),imshow(uint8(simg));
title({'Fig.4.18(a)钝化模板';'(原图减低通)'})
subplot(133),imshow(uint8(simg1));
title({'Fig.4.26(a)钝化模板';'(高通)'})

% (b)
simg2 = I - gauss(I,80,1);
simg22 = gauss(I,80,0);
figure,
subplot(131),imshow(uint8(I));
title('Fig.4.18(a)原图')
subplot(132),imshow(uint8(simg2));
title({'Fig.4.18(a)钝化模板';'(原图减低通)'})
subplot(133),imshow(uint8(simg22));
title({'Fig.4.26(c)钝化模板';'(高通)'})

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
% rel = conj(f1).* f2;
rel = f2 .* conj(f1);
newI = recover(ifft2(rel));
figure,
subplot(131),imshow(uint8(I1));
title('Fig.4.41(a)原图')
subplot(132),imshow(uint8(I2));
title('Fig.4.41(b)原图')
subplot(133),imshow(mat2gray(newI),[]);
title('Fig.4.41图像相关')
max_value = max(max(newI));
[row,col] = find(newI == max_value)

% Fig 4.04(a)
Ir = imread('Fig0404(a).jpg');
I0 = frotate(Ir,0);
I1 = frotate(Ir,45);
I2 = frotate(Ir,90);
I3 = frotate(Ir,135);
I4 = frotate(Ir,180);
figure,
subplot(231),imshow(I);
title('Fig.4.04(a)原图')
subplot(232),imshow(log(I0 +1));
title('原图傅里叶谱')
subplot(233),imshow(log(I4 +1));
title('旋转180°傅里叶谱')
subplot(234),imshow(log(I1 +1));
title('旋转45°傅里叶谱')
subplot(235),imshow(log(I3 +1));
title('旋转135°傅里叶谱')
subplot(236),imshow(log(I2 +1));
title('旋转90°傅里叶谱')

% PROJECT 04-03 (a)
function g = gauss(img,sig,lowpass_flag)
	[M,N] = size(img);
	P = 2 * M; Q = 2 * N; % remember to do extension
	Iext = zeros(P,Q);
	Iext(1:M,1:N) = img(1:M,1:N);
	[Y,X] = meshgrid(1:Q,1:P);
    center_x = P/2; center_y = Q/2;
	D = (X - center_x).^2 + (Y - center_y).^2;
	if lowpass_flag == 1
		H = exp(-D/(2*sig^2));
	else
		H = 1 - exp(-D/(2*sig^2));
    end
	cimg = centerize(Iext);
	f = fft2(cimg);
	g = centerize(real(ifft2(H.*f)));
	g = g(1:M,1:N);
end

% PROJECT 04-01
% (a)
function g = centerize(img)
	[M,N] = size(img);
	[Y,X] = meshgrid(1:N,1:M);
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

% rotate
function g = frotate(img,ang)
	rI = imrotate(img,ang);
	FI = ifft2(centerize(double(rI)));
	g = abs(FI);
end