function output = pf(PIPELINE_LENGTH, PIPELINE_SIZE, H, combinedMasks)
    fprintf('Running the Pipeline Filter...\n');
    startTime = tic;

    % Buffer to store recent frame data
    buffer = cell(PIPELINE_SIZE + 1, 1);

    % Initialize output storage for confirmed objects
    output = [];
    totalFrames = numel(combinedMasks);

    % Iterate over the video frames
    for currentFrame = 1:totalFrames
        endFrame = min(currentFrame + PIPELINE_LENGTH, totalFrames);

        if currentFrame == 1 % Fill the buffer initially
            for bufferIdx = currentFrame:endFrame
                if bufferIdx > totalFrames
                    buffer{bufferIdx - currentFrame + 1} = computeFrameData(zeros(size(combinedMasks, 1), size(combinedMasks, 2)));
                else
                    buffer{bufferIdx - currentFrame + 1} = computeFrameData(combinedMasks{bufferIdx});
                end
            end
        end

        % Print out all bounding boxes in buffer for debugging (first frame in buffer)
        for i = 1:numel(buffer{1})
            fprintf('Frame %d: Object %d: x=%d, y=%d, width=%d, height=%d\n', currentFrame, i, buffer{1}(i).x, buffer{1}(i).y, buffer{1}(i).width, buffer{1}(i).height);
        end



        % Shift the buffer
        buffer(1:end-1) = buffer(2:end);

        if endFrame + 1 <= totalFrames
            buffer{end} = computeFrameData(combinedMasks{endFrame + 1});
        else
            buffer{end} = computeFrameData(zeros(size(combinedMasks, 1), size(combinedMasks, 2)));
        end
    end
end

function frameData = computeFrameData(frame)
    labeledImage = bwlabel(frame);
    props = regionprops(labeledImage, 'BoundingBox');

    if isempty(props)
        % Initialize with default values if no regions are found
        frameData = struct('id', {}, 'x', {}, 'y', {}, 'width', {}, 'height', {});
    else
        % Convert props to a structured array
        numProps = numel(props);
        frameData(numProps).id = [];
        frameData(numProps).x = [];
        frameData(numProps).y = [];
        frameData(numProps).width = [];
        frameData(numProps).height = [];
        for propIdx = 1:numProps
            bb = props(propIdx).BoundingBox;
            frameData(propIdx).id = -1;
            frameData(propIdx).x = bb(1);
            frameData(propIdx).y = bb(2);
            frameData(propIdx).width = bb(3);
            frameData(propIdx).height = bb(4);
        end
    end
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
