function [ output_args ] = Meanface( T)
%MEANFACE Summary of this function goes here
%   Detailed explanation goes here
T=mean(T,2);
face=reshape(T,180,200);
face=uint8(face);
figure
imshow(face);
output_args=0;
end

