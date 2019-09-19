close all;clear all;clc;
I=imread('Fig0340.tif');
[m,n]=size(I);
newI=zeros(m,n);
for i = 2:m-1
    for j = 2:n-1
        newI(i,j)=-I(i+1,j)-I(i-1,j)-I(i,j+1)-I(i,j-1)+8*I(i,j)-I(i+1,j+1)-I(i+1,j-1)-I(i-1,j+1)-I(i-1,j-1);
    end
end
figure,
subplot(121),imshow(uint8(I));
title('原图')
subplot(122),imshow(uint8(newI));
title('Laplacian变换后');

sigma=3;
gauss=zeros(5,5);
sumup=0;
for i = -2:2
	for j = -2:2
		gauss(i+3,j+3)=exp(-(i*i+j*j)/(2*sigma*sigma));
        sumup = sumup + gauss(i+3,j+3); % normalization
	end
end
gauss=gauss./sumup;
newIb=zeros(m,n);
for i = 3:m-2
	for j = 3:n-2
		for u = -2:2
			for v = -2:2
				newIb(i,j) = newIb(i,j) + uint8(gauss(u+3,v+3)*I(i+u,j+v));
			end
		end
	end
end

newIc=zeros(m,n);
for i = 2:m-1
	for j = 2:n-1
		avg = 1/9*(I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)+I(i,j)+I(i+1,j+1)+I(i+1,j-1)+I(i-1,j+1)+I(i-1,j-1));
		newIc(i,j)=I(i,j)-uint8(avg);
	end
end

newId=zeros(m,n);
for i = 2:m-1
	for j = 2:n-1
		avg = 1/9*(I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)+I(i,j)+I(i+1,j+1)+I(i+1,j-1)+I(i-1,j+1)+I(i-1,j-1));
		newId(i,j)=2*I(i,j)-uint8(avg);
	end
end

newIe=zeros(m,n);
for i = 2:m-1
	for j = 2:n-1
		avg = 1/9*(I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)+I(i,j)+I(i+1,j+1)+I(i+1,j-1)+I(i-1,j+1)+I(i-1,j-1));
		newIe(i,j)=I(i,j)+4.5*(I(i,j)-uint8(avg));
	end
end

figure,
subplot(151),imshow(uint8(I));
title('3.40(a)原图')
subplot(152),imshow(uint8(newIb));
title('高斯滤波');
subplot(153),imshow(uint8(newIc));
title('非锐化模板');
subplot(154),imshow(uint8(newId));
title('非锐化掩盖');
subplot(155),imshow(uint8(newIe));
title('高提升滤波');