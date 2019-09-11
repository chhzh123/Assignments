close all;clear all;clc;
I=imread('Fig0308.tif');
[m,n]=size(I);
% J=histeq(I);
A=zeros(1,256);
for i = 1:256
    A(i)=sum(sum(I == (i-1))); % eliminate zeros
end
A=double(A);
A=A./(m*n); % normalization
cumulation=zeros(1,256);
for i = 1:256
    for j = 1:i
        cumulation(i)=cumulation(i)+A(j);
    end
end
newI=zeros(m,n);
for i = 1:m
    for j = 1:n
        newI(i,j)=uint8(cumulation(I(i,j)+1)*255);
    end
end
newA=zeros(1,256);
for i = 1:256
    newA(i)=sum(sum(newI == (i-1))); % eliminate zeros
end
figure,
subplot(121),imshow(uint8(I));
title('原图')
subplot(122),imshow(uint8(newI));
title('均衡化后')
figure,
% subplot(121),imhist(I,64);
subplot(121),bar(0:255,uint32(A*255));
title('原图像直方图');
% subplot(122),imhist(newI,64);
subplot(122),bar(0:255,uint32(newA));
title('均衡化后的直方图');