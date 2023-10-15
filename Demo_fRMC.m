clc;
close all;
clear variables;

%% Initialize parameters
tElapsed = nan;
Disp = 0;
max_niter = 10;
gamma1 = 0.8;
gamma2 = 0.8;
L = 4;
f = 30;
savePathSt = fullfile('processing/', 'lrmc/');

dir_path = './processing/frames/';

files = dir(fullfile(dir_path, '*.bmp'));
M = length(files);

N = floor(M / (L * f));

frameCounter = 1;

while frameCounter < M
    imArray = [];

    offset = 0;

    if frameCounter ~= 1
        offset = floor(M/(2*N));
    end

    frameCounter = frameCounter - offset;

    inner_loop = min(floor(M/N), M - frameCounter + 1);

    for j = 1:inner_loop
        idx = frameCounter + j - 1; 
        img_path = fullfile(dir_path, [num2str(idx) '.bmp']);

        if ~isfile(img_path)
            break;
        end

        imArray(:,:,j) = rgb2gray(imread(img_path));
    end


    if isempty(imArray) || ndims(imArray) < 3
        break;
    end

    %% Extracting the size information of the cropped images and reshaping the sequence
    orgSize = size(imArray);
    imNum = orgSize(3);   % number of images in the dataset
    imDim = orgSize(1:2); % resolution of each image
    rate = 1;   % down sampling rate
    height = imDim(1)/rate;
    width = imDim(2)/rate;
    dwnSize = height*width/(rate^2);  % dimension of the downsampled image as a vector

    %% Applying the fRMC on gray scaled images
    imMatG = reshape(imArray, height*width, []);
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

    for j = 1:size(ForegMask, 3)
        FileName = strcat(num2str(frameCounter), '.bmp'); 
        path = fullfile(savePath, FileName);
        imwrite(ForegMask(:, :, j), path);
        frameCounter = frameCounter + 1;
    end
end 