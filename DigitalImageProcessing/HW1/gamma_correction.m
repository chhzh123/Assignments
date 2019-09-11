close all;clear all;clc;
GAMMA = 2.5
I=imread('Fig0308.tif');
[m,n]=size(I);
newI=zeros(m,n);
I=double(I);
for i = 1:m
    for j = 1:n
        newI(i,j)=I(i,j).^GAMMA;
    end
end
newI=(newI-min(min(newI)))/(max(max(newI))-min(min(newI)))*255;
figure,
subplot(121),imshow(uint8(I));
title('原图')
subplot(122),imshow(uint8(newI));
title('伽马校正后(gamma=2.5)')