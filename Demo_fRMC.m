function Demo_fRMC(max_niter_param, gamma2_param, current_frame_param, N_param)
    max_niter = max_niter_param;
    gamma2 = gamma2_param;
    current_frame = current_frame_param;
    N = N_param;
    L = L_param;

    savePathSt = fullfile('processing/', 'lrmc/');

    dir_path = './processing/frames/';

    imArray = [];

    for i = 1:N
        idx = current_frame_param + i - 1;
        img_path = fullfile(dir_path, [num2str(idx) '.bmp']);

        if ~exist(img_path, 'file')
            break;
        end

        imArray(:,:,i) = rgb2gray(imread(img_path));
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
    [A, ~ ] = InfaceExtFrankWolfe(imMatG, [], gamma2, max_niter);
    E = abs(A - imMatG);
    savePath = fullfile(savePathSt);
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

    FileName = strcat(num2str(current_frame_param), '.bmp'); 
    path = fullfile(savePath, FileName);
    imwrite(ForegMask(:, :, 1), path);
end