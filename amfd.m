function output = amfd(K, CONNECTIVITY, AREA_MIN, AREA_MAX, ASPECT_RATIO_MIN, ASPECT_RATIO_MAX, KERNEL, imageSequence)
    fprintf('Processing frames using AMFD...\n\n');
    startTime = tic;
    numFrames = numel(imageSequence);
    output = cell(1, numFrames);
    output{1} = zeros(size(imageSequence{1}, 1), size(imageSequence{1}, 2), 'uint8');  % Handle edge frames
    output{numFrames} = zeros(size(imageSequence{end}, 1), size(imageSequence{end}, 2), 'uint8');

    % Preallocate and reuse the structuring element
    se = createDisk(KERNEL);

    for t = 2:(numFrames - 1)
        % Convert images to grayscale once and store for reuse
        if t == 2  % Initialize previous and current images
            I_prev = double(rgb2gray(imageSequence{t-1}));
            I_curr = double(rgb2gray(imageSequence{t}));
        else  % Shift images and read next only
            I_prev = I_curr;
            I_curr = I_next;
        end
        I_next = double(rgb2gray(imageSequence{t+1}));

        % Calculate difference images and accumulate
        D_t1 = abs(I_curr - I_prev);
        D_t2 = abs(I_next - I_prev);
        D_t3 = abs(I_next - I_curr);
        Id = (D_t1 + D_t2 + D_t3) / 3;

        % Threshold calculation
        mu = mean(Id(:));
        sigma = std(Id(:));
        T = mu + K * sigma;

        % Binarize and morphological operations
        binaryImage = imclose(imopen(Id >= T, se), se);

        % Connected components and properties
        labeledImage = bwlabel(binaryImage, CONNECTIVITY);
        props = regionprops(labeledImage, 'Area', 'BoundingBox');
        for propIdx = 1:length(props)
            bb = props(propIdx).BoundingBox;
            aspectRatio = bb(3) / bb(4);
            if props(propIdx).Area < AREA_MIN || props(propIdx).Area > AREA_MAX || aspectRatio < ASPECT_RATIO_MIN || aspectRatio > ASPECT_RATIO_MAX
                binaryImage(labeledImage == propIdx) = 0;
            end
        end

        % Store the refined binary mask
        output{t} = binaryImage;
        printProgressBar(t, numFrames - 1, startTime);  % Update progress bar
    end
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