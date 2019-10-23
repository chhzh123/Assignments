close all;clear all;clc;

I = imread('fig/Fig0516.tif');
I = double(I);
newI = filter(I);

figure,
subplot(221),imshow(uint8(I));
title('Fig.5.16(a)原图')
subplot(222),imshow(uint8(abs(fft2(centerize(I))).^0.4),[]); % (log(1 + sp),[]);
title('Fig.5.16(a)傅里叶谱')
subplot(223),imshow(uint8(newI));
title('带阻滤波后结果')
subplot(224),imshow(uint8(abs(fft2(centerize(newI))).^0.4),[]); % (log(1 + sp),[]);
title('带阻滤波后傅里叶谱')

function g = filter(img)
	[M,N] = size(img);
	P = 2 * M; Q = 2 * N; % remember to do extension
	Iext = zeros(P,Q);
	Iext(1:M,1:N) = img(1:M,1:N);
	[Y,X] = meshgrid(1:Q,1:P);
    center_x = P/2; center_y = Q/2;
	D = (X - center_x).^2 + (Y - center_y).^2;
	D0 = 300^2;
	W = 400;
	n = 1;
%     H = 1 - exp(-0.5*(D-55^2)/((sqrt(D)*5).^2));
	H = 1 ./ (1 + ((sqrt(D) * W)./(D - D0)).^(2*n));
	cimg = centerize(Iext);
	f = fft2(cimg);
	g = centerize(real(ifft2(H.*f)));
	g = g(1:M,1:N);
end

function g = centerize(img)
	[M,N] = size(img);
	[Y,X] = meshgrid(1:N,1:M);
	ones = (-1).^(X+Y);
	g = ones.*img;
end