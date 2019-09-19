close all;clear all;clc;
I=imread('Fig0343.tif');
[m,n]=size(I);
newI=zeros(m,n);
A=2;
for i = 2:m-1
	for j = 2:n-1
		avg = 1/9*(I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)+I(i,j)+I(i+1,j+1)+I(i+1,j-1)+I(i-1,j+1)+I(i-1,j-1));
		newI(i,j)=I(i,j)+A*(I(i,j)-uint8(avg));
	end
end
figure,
subplot(121),imshow(uint8(I));
title('3.43(a)原图')
subplot(122),imshow(uint8(newI));
title('高提升滤波后(A=2)');