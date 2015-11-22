clear;clc 
AA=imread('./16.jpg');
% imshow(AA); 
for k=1:3 
BB(:,:,k)=flipud(AA(:,:,k)); %up-down
B(:,:,k)=fliplr(AA(:,:,k));  %left-right
end 
% figure; 
% imshow(BB); 
imwrite(BB, '18.jpg');
% figure
% imshow(B); 
imwrite(B, '19.jpg');