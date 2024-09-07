function output = lrmc(L, KERNEL, MAX_NITER_PARAM, GAMMA1_PARAM, GAMMA2_PARAM, FRAME_RATE, grayFrames, USE_PARALLEL_LRMC, NUM_WORKERS)
% fprintf('Processing frames using LRMC...\n');
numFrames = numel(grayFrames);
N = max(1, min(floor(numFrames / (L * FRAME_RATE)), numFrames));  % Ensure N is at least 1

output = cell(1, numFrames);  % Preallocate output cell array
se = strel('disk', double(max(1, floor(KERNEL/2))));

if USE_PARALLEL_LRMC
    % Respawn workers
    delete(gcp('nocreate'));
    parpool('local', NUM_WORKERS);

    % Use the specified number of workers
    numWorkers = NUM_WORKERS;
    chunkSize = ceil(numFrames / numWorkers);
    
    parforOutput = cell(1, numWorkers);
    
    parfor workerIdx = 1:numWorkers
        startIdx = (workerIdx - 1) * chunkSize + 1;
        endIdx = min(workerIdx * chunkSize, numFrames);
        workerOutput = cell(1, endIdx - startIdx + 1);
        
        for frameIdx = startIdx:endIdx
            imArray = [];
            
            % Collect frames
            for j = 1:N
                idx = frameIdx + j - 1;
                if idx > numFrames
                    break;
                end
                imArray(:, :, j) = grayFrames{idx};
            end
            
            % Check if imArray is empty or has fewer than 2 frames
            if isempty(imArray) || size(imArray, 3) < 2
                workerOutput{frameIdx - startIdx + 1} = zeros(size(imArray(:, :, 1)), 'uint8');
                continue;
            end
            
            % Surpress warnings
            warnState = warning('off', 'all');
            
            % Process the frame and save the mask
            workerOutput{frameIdx - startIdx + 1} = processFrame(imArray, GAMMA1_PARAM, GAMMA2_PARAM, MAX_NITER_PARAM, se);
            
            warning(warnState);
        end
        
        parforOutput{workerIdx} = workerOutput;
    end
    
    % Combine results after parfor loop
    output = [parforOutput{:}];
else
    for frameIdx = 1:numFrames
        imArray = [];
        
        % Collect frames
        for j = 1:N
            idx = frameIdx + j - 1;
            if idx > numFrames
                break;  % If the index is out of bounds, break the loop
            end
            imArray(:, :, j) = grayFrames{idx};
        end
        
        % Check if imArray is empty or has fewer than 2 frames
        if isempty(imArray) || size(imArray, 3) < 2
            output{frameIdx} = zeros(size(imArray(:, :, 1)), 'uint8');
            continue;
        end
        
        % Surpress warnings
        warnState = warning('off', 'all');
        
        % Process the frame and save the mask
        output{frameIdx} = processFrame(imArray, GAMMA1_PARAM, GAMMA2_PARAM, MAX_NITER_PARAM, se);
        
        warning(warnState);
    end
end
end

function mask = processFrame(imArray, gamma1, gamma2, max_niter, kernel)
imDim = size(imArray, [1, 2]);
dwnSize = prod(imDim);

imMatG = reshape(double(imArray), dwnSize, []);
[A, ~] = InfaceExtFrankWolfe(imMatG, gamma1, gamma2, max_niter);
E = abs(A - imMatG);

Th = (1 / 5) * max(E(:));
ForegMask = E > Th;
ForegMask = reshape(ForegMask, [imDim, size(imArray, 3)]);
ForegMask = ForegMask(:, :, 1);
ForegMask = imopen(ForegMask, kernel);
ForegMask = imclose(ForegMask, kernel);
mask = imfill(ForegMask, 'holes');
end