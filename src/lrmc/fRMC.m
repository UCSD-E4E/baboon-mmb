function fRMC(max_niter, gamma1, gamma2, N, frame_count, kernel)
    savePathSt = './processing/lrmc/';
    dir_path = './processing/frames/';
    
    % Define structuring elements outside of the loop for efficiency
    rectangleKernel = strel('rectangle', [kernel, kernel]);
    
    % Precompute static values if they don't change between iterations
    
    % Use a parallel for loop to distribute iterations to different workers
    parfor i = 1:frame_count
        current_frame = i; % Adjusted to match the parallel loop variable
        imArray = [];
        
        % Read frames
        for j = 1:N
            idx = current_frame + j - 1;
            img_path = fullfile(dir_path, sprintf('%d.bmp', idx));

            if ~exist(img_path, 'file')
                continue; % If the file does not exist, just continue to the next iteration
            end
            
            imArray(:, :, j) = rgb2gray(imread(img_path));
        end

        % If there are not enough frames, continue to the next iteration
        if size(imArray, 3) <= 1
            % Calculate imDim here, since imArray has at least one frame
            imDim = size(imArray(:, :, 1));
            path = fullfile(savePathSt, sprintf('%d.bmp', current_frame));
            imwrite(zeros(imDim), path); % Using imDim directly here
            continue;
        end
        
        % Process and save the frame
        processAndSaveFrame(imArray, current_frame, savePathSt, gamma1, gamma2, max_niter, rectangleKernel);
    end
end

function processAndSaveFrame(imArray, current_frame, savePathSt, gamma1, gamma2, max_niter, rectangleKernel)
    imDim = size(imArray(:,:,1));
    dwnSize = prod(imDim);

    imMatG = reshape(double(imArray), dwnSize, []);
    [A, ~] = InfaceExtFrankWolfe(imMatG, gamma1, gamma2, max_niter);
    E = abs(A - imMatG);
    
    Th = (1/5) * max(E(:));
    ForegMask = E > Th;
    ForegMask = reshape(ForegMask, [imDim, size(imArray, 3)]);
    ForegMask = ForegMask(:, :, 1);
    ForegMask = imopen(ForegMask, rectangleKernel);
    ForegMask = imclose(ForegMask, rectangleKernel);
    ForegMask = imfill(ForegMask, 'holes');
    ForegMask = uint8(ForegMask) * 255;

    path = sprintf('%s/%d.bmp', savePathSt, current_frame);
    imwrite(ForegMask, path);
end