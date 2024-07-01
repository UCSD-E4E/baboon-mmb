function output = lrmc(L, KERNEL, MAX_NITER_PARAM, GAMMA1_PARAM, GAMMA2_PARAM, FRAME_RATE, imageSequence)
    fprintf('Processing frames using LRMC...\n');
    startTime = tic;
    numFrames = numel(imageSequence);
    N = max(1, floor(numFrames / (L * FRAME_RATE))); % Ensure N is at least 1

    output = cell(1, numFrames);  % Preallocate output cell array
    se = createDisk(KERNEL);

    oldWarningState = warning('off', 'Octave:divide-by-zero');

    for i = 1:numFrames
        imArray = [];

        % Collect frames
        for j = 1:N
            idx = i + j - 1;
            if idx > numFrames
                break;  % If the index is out of bounds, break the loop
            end
            imArray(:, :, j) = rgb2gray(imageSequence{idx});
        end 

        % If there are not enough frames, continue to the next iteration
        if size(imArray, 3) < 2
            output{i} = zeros(size(imArray(:, :, 1)), 'uint8');
            continue;
        end

        % Process the frame and save the mask
        output{i} = processFrame(imArray, GAMMA1_PARAM, GAMMA2_PARAM, MAX_NITER_PARAM, se);
        printProgressBar(i, numFrames, startTime);  % Update progress bar
    end

    warning(oldWarningState);  % Restore the warning state
end

function mask = processFrame(imArray, gamma1, gamma2, max_niter, kernel)
    imDim = size(imArray(:,:,1));
    dwnSize = prod(imDim);

    imMatG = reshape(double(imArray), dwnSize, []);
    [A, ~] = InfaceExtFrankWolfe(imMatG, gamma1, gamma2, max_niter);
    E = abs(A - imMatG);

    Th = (1/5) * max(E(:));
    ForegMask = E > Th;
    ForegMask = reshape(ForegMask, [imDim, size(imArray, 3)]);
    ForegMask = ForegMask(:, :, 1);
    ForegMask = imopen(ForegMask, kernel);
    ForegMask = imclose(ForegMask, kernel);
    mask = imfill(ForegMask, 'holes');
end

function se = createDisk(diameter)
    radius = floor(diameter / 2);  % Calculate the radius from the diameter
    N = 2 * radius + 1;  % The side length of the smallest square that can contain the circle
    [x, y] = meshgrid(-radius:radius, -radius:radius);
    mask = (x.^2 + y.^2) <= radius^2;  % Create a logical mask where points are within the radius
    se = strel('arbitrary', mask);  % Create the structuring element from the logical mask
end

function printProgressBar(currentStep, totalSteps, startTime)
    % Calculate percentage completion
    percentage = 100 * (currentStep / totalSteps);
    barLength = floor(50 * (currentStep / totalSteps));  % Length of the progress bar in characters
    bar = repmat('#', 1, barLength);  % Create the progress bar
    spaces = repmat(' ', 1, 50 - barLength);  % Spaces to fill the rest of the bar
    
    % Calculate elapsed time and estimate remaining time
    elapsedTime = toc(startTime);
    remainingTime = elapsedTime / currentStep * (totalSteps - currentStep);
    
    % Format remaining time as HH:MM:SS
    hours = floor(remainingTime / 3600);
    mins = floor(mod(remainingTime, 3600) / 60);
    secs = floor(mod(remainingTime, 60));
    
    % Clear the previous line before printing new progress information
    if currentStep > 1  % Avoid clearing if it's the first step
        fprintf('\033[A\033[K');  % Move cursor up one line and clear line
    end
    
    % Print progress bar with time estimate
    fprintf('[%s%s] %3.0f%% - Elapsed: %02d:%02d:%02d, Remaining: %02d:%02d:%02d\n', ...
            bar, spaces, percentage, ...
            floor(elapsedTime / 3600), mod(floor(elapsedTime / 60), 60), mod(elapsedTime, 60), ...
            hours, mins, secs);

    if currentStep == totalSteps
        fprintf('\n');  % Move to the next line after completion
    end
end