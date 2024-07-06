function output = lrmc(L, KERNEL, MAX_NITER_PARAM, GAMMA1_PARAM, GAMMA2_PARAM, FRAME_RATE, grayFrames)
    fprintf('Processing frames using LRMC...\n');
    numFrames = numel(grayFrames);
    N = max(1, floor(numFrames / (L * FRAME_RATE)));  % Ensure N is at least 1

    output = cell(1, numFrames);  % Preallocate output cell array
    se = strel('disk', double(max(1, floor(KERNEL/2))));

    % Broadcast grayFrames to workers
    grayFrames = parallel.pool.Constant(grayFrames);

    parfor frameIdx = 1:numFrames
        imArray = [];

        % Collect frames
        for j = 1:N
            idx = frameIdx + j - 1;
            if idx > numFrames
                break;  % If the index is out of bounds, break the loop
            end
            imArray(:, :, j) = grayFrames.Value{idx};
        end 

        % If there are not enough frames, continue to the next iteration
        if size(imArray, 3) < 2
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