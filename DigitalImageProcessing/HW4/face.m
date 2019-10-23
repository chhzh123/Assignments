close all;clear all;clc;

I = imread('fig/TestImage.jpg');

BW_rgb = rgb_face(I);
BW_hsv = hsv_face(I);
BW_ycbcr = ycbcr_face(I);

bb_rgb = get_bb(BW_rgb);
bb_hsv = get_bb(BW_hsv);
bb_ycbcr = get_bb(BW_ycbcr);

figure, subplot(1,2,2);
imshow(BW_rgb)
title('人脸部分标识');
subplot(1,2,1);
title('人脸识别结果');
imshow(I);
hold on;
rectangle('Position',bb_rgb,'EdgeColor','r')

figure, subplot(1,2,2);
imshow(BW_hsv)
title('人脸部分标识');
subplot(1,2,1);
title('人脸识别结果');
imshow(I);
hold on;
rectangle('Position',bb_hsv,'EdgeColor','r')

figure, subplot(1,2,2);
imshow(BW_ycbcr)
title('人脸部分标识');
subplot(1,2,1);
title('人脸识别结果');
imshow(I);
hold on;
rectangle('Position',bb_ycbcr,'EdgeColor','r')

function bw = rgb_face(I)
	[m,n,c] = size(I);
	BW = zeros(m,n);
	for i = 1:size(I,1)
		for j = 1:size(I,2)
			R = I(i,j,1);
			G = I(i,j,2);
			B = I(i,j,3);
			v = [R,G,B];
			if (R > 95 && G > 40 && B > 20 && (max(v) - min(v)) > 15 && abs(R-G) > 15 && R > G && R > B) % day
			% if (R > 20 && G > 210 && B > 170 && abs(R-G) < 15 && R > G && R > B) % night
				BW(i,j) = 1;
			end
		end
	end
	bw = BW;
end

function bw = hsv_face(I)
	I_h = rgb2hsv(I);
	[m,n,c] = size(I_h);
	BW = zeros(m,n);
	for i = 1:m
		for j = 1:n
			h = I_h(i,j,1);
			s = I_h(i,j,2);
			v = I_h(i,j,3);
			if (v > 40 && s >= 0.2 && s <= 0.6 && h >= 0 && h <= 0.25)
				BW(i,j) = 1;
			end
		end
	end
	bw = BW;
end

function bw = ycbcr_face(I)
	I_y = rgb2ycbcr(I);
	[m,n,c] = size(I_y);
	BW = zeros(m,n);
	for i = 1:m
		for j = 1:n
			y = I_y(i,j,1);
			cb = I_y(i,j,2);
			cr = I_y(i,j,3);
			if (75 < cb && cb < 250 && 10 < cr && cr < 100 && y > 80)
				BW(i,j) = 1;
			end
		end
	end
	bw = BW;
end

function [a,b,c,d] = get_bb(BW)
	L = bwlabel(BW,8); % 8 connectivity
	% Left Top Width Height
	BB = regionprops(L,'BoundingBox'); % get smallest retangle, return as a structure
	% xMin = ceil(BoundingBox(1))
	% xMax = xMin + BoundingBox(3) - 1
	% yMin = ceil(BoundingBox(2))
	% yMax = yMin + BoundingBox(4) - 1
	BB1 = struct2cell(BB); % struct to cell
	BB2 = cell2mat(BB1); % cell to matrix

	[s1 s2] = size(BB2);
	mx = 0;
	for k = 3:4:s2-1
	    area_bb = BB2(1,k) * BB2(1,k+1);
	    if p > max_area && (BB2(1,k) / BB2(1,k+1)) < 1.8
	        max_area = p;
	        j = k;
	    end
	end
	[a,b,c,d] = [BB2(1,j-2),BB2(1,j-1),BB2(1,j),BB2(1,j+1)]
end