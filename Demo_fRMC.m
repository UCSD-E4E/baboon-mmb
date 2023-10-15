clc;
close all;
clear variables;

%% Initialize parameters
tElapsed = nan;
Disp = 0;
max_niter = 10;
gamma1 = 0.8;
gamma2 = 0.8;
savePathSt = fullfile('processing/', '.lrmc/');

%% Reading images sequentially from ../processing/frames and save it as a matrix
i = 1;
while true
    img_path = fullfile('./processing/frames', [num2str(i) '.bmp']);
    if ~isfile(img_path)
        break;
    end
    %% Saves grey intensity to 3d array. i = frame#, (:,:) = (height, width)
    imArray(:,:,i) = rgb2gray(imread(img_path));
    i = i + 1;
end

%% Extracting the size information of the cropped images and reshaping the sequence
orgSize = size(imArray);
imNum = orgSize(3);   % number of images in the dataset
imDim = orgSize(1:2); % resolution of each image
rate = 1;   % down sampling rate
height = imDim(1)/rate;
width = imDim(2)/rate;
dwnSize = height*width/(rate^2);  % dimension of the downsampled image as a vector

%% LRMC Optimization
L = 4
N = i / (L * 10)

%% Applying the fRMC on gray scaled images
%% Reshape (height, width, frame) to (height*width, frame)
newImMatG = reshape(imArray, height*width, []);
imMatG = movmean(newImMatG,N,2,"Endpoints","discard");
clear imArray
imMatG = double(imMatG);
frNum = size(imMatG, 2);
tic;
[A, ~ ] = InfaceExtFrankWolfe(imMatG, [], gamma2, max_niter);
tElapsed = toc;
E = abs(A - imMatG);
savePath = fullfile(savePathSt);
save(fullfile(savePath,'Foreground.mat'), 'E');
save(fullfile(savePath,'Background.mat'), 'A');
clear A

%% Save the binary mask as video
power = 1;
coef = 1;
ForegEn = coef * E .^ power;
Th = (1/5) * max(max(ForegEn));
ForegMask = ForegEn > Th;
ForegMask = reshape(ForegMask, height, width, []);
ForegMask = imopen(ForegMask, strel('rectangle', [3,3]));
ForegMask = imclose(ForegMask, strel('rectangle', [5, 5]));
ForegMask = imfill(ForegMask, 'holes');
ForegMask = 255* uint8(ForegMask);
v = VideoWriter(fullfile(savePath, 'forground.avi'), 'Grayscale AVI');
v.FrameRate = 10;
open(v);
writeVideo(v, ForegMask);
close(v);

for j = 1:size(ForegMask, 3)
    FileName = strcat('fg_', num2str(j, '%.06i'), '.png');
    path = fullfile(savePath, FileName);
    imwrite(ForegMask(:, :, j), path);
end

save(fullfile(savePath, 'elapse-time.txt'), 'tElapsed', '-ascii');
