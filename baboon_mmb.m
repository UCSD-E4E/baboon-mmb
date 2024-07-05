function baboon_mmb(varargin)
    p = inputParser;

    addParameter(p, 'K', 4, @(x) x >= 0 && x <= 8);
    addParameter(p, 'CONNECTIVITY', 8, @(x) any(x == [4, 8]));
    addParameter(p, 'AREA_MIN', 5, @(x) x >= 0);
    addParameter(p, 'AREA_MAX', 80, @(x) x > 5 && x <= 100);
    addParameter(p, 'ASPECT_RATIO_MIN', 1.0, @(x) x >= 0);
    addParameter(p, 'ASPECT_RATIO_MAX', 6.0, @(x) x > 1 && x <= 10);
    addParameter(p, 'L', 4, @(x) x >= 1 && x <= 10);
    addParameter(p, 'KERNEL', 3, @(x) any(x == [1, 3, 5, 7, 9, 11]));
    addParameter(p, 'PIPELINE_LENGTH', 5, @(x) x >= 1 && x <= 10);
    addParameter(p, 'PIPELINE_SIZE', 7, @(x) any(x == [3, 5, 7, 9, 11]));
    addParameter(p, 'H', 3, @(x) x >= 1);
    addParameter(p, 'MAX_NITER_PARAM', 10, @(x) x >= 1 && x <= 20);
    addParameter(p, 'GAMMA1_PARAM', 0.8, @(x) x >= 0.0 && x <= 1.0);
    addParameter(p, 'GAMMA2_PARAM', 0.8, @(x) x >= 0.8 && x <= 1.0);
    addParameter(p, 'FRAME_RATE', 10, @(x) x >= 1);
    addParameter(p, 'IMAGE_SEQUENCE', '', @ischar);

    parse(p, varargin{:});
    args = p.Results;

    if args.ASPECT_RATIO_MIN >= args.ASPECT_RATIO_MAX
        error('ASPECT_RATIO_MIN must be less than ASPECT_RATIO_MAX');
    end

    if args.GAMMA1_PARAM > args.GAMMA2_PARAM
        error('GAMMA1_PARAM must be less than GAMMA2_PARAM');
    end

    % imageSequence = loadImageSequence(args.IMAGE_SEQUENCE);

    % amfdMasks = amfd(args.K, args.CONNECTIVITY, args.AREA_MIN, args.AREA_MAX, args.ASPECT_RATIO_MIN, args.ASPECT_RATIO_MAX, args.KERNEL, imageSequence);
    % saveMasks(amfdMasks, 'output/amfd');
    % save('output/amfdMasks.mat', 'amfdMasks');
    % clear amfdMasks;

    % lrmcMasks = lrmc(args.L, args.KERNEL, args.MAX_NITER_PARAM, args.GAMMA1_PARAM, args.GAMMA2_PARAM, args.FRAME_RATE, imageSequence);
    % saveMasks(lrmcMasks, 'output/lrmc');
    % save('output/lrmcMasks.mat', 'lrmcMasks');
    % clear lrmcMasks;
    
    % load('output/amfdMasks.mat', 'amfdMasks');
    % load('output/lrmcMasks.mat', 'lrmcMasks');
    % combinedMasks = combineMasks(amfdMasks, lrmcMasks); 
    % saveMasks(combinedMasks, 'output/combinedMasks');
    % save('output/combinedMasks.mat', 'combinedMasks');
    % clear amfdMasks lrmcMasks combinedMasks;
   
    load 'output/combinedMasks.mat';
    objects = pf(args.PIPELINE_LENGTH, args.PIPELINE_SIZE, args.H, combinedMasks);
    % save('output/objects.mat', 'objects');

    % saveObjectsToTxt(objects, 'output/objects.txt');

    % drawBoundingBoxesOnFrames(imageSequence, objects, 'output/frames');
end 

function imageSequence = loadImageSequence(imagePath)
    fprintf('Loading image sequence...\n', imagePath);
    startTime = tic;
    if ~isempty(imagePath)
        files = dir(fullfile(imagePath, '*.jpg'));
        [~, idx] = sort({files.name});
        files = files(idx);
        imageSequence = cell(1, numel(files));
        for fileIdx = 1:numel(files)
            imageSequence{fileIdx} = imread(fullfile(imagePath, files(fileIdx).name));
            printProgressBar(fileIdx, numel(files), startTime);  % Call to function that prints the progress bar
        end
    else
        error('Image sequence folder is not specified or does not exist.');
    end
end

function saveMasks(output, outputDir)
    fprintf('Saving masks...\n');
    startTime = tic;
    % Ensure the output directory exists
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Loop through each frame in the output
    numFrames = numel(output);
    for frameIdx = 1:numFrames
        if ~isempty(output{frameIdx}) % Ensure there is data to write
            filename = fullfile(outputDir, sprintf('%06d.png', frameIdx));
            imwrite(output{frameIdx}, filename, 'png');
        end
        printProgressBar(frameIdx, numFrames, startTime);  % Call to function that prints the progress bar
    end
end

function drawBoundingBoxesOnFrames(imageSequence, objects, outputFolder)
    fprintf('Sorting bounding boxes...\n');
    startTime = tic;

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    frameObjects = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    
    for objectIdx = 1:numel(objects)
        frameIdx = objects{objectIdx}{1};
        bbox = [objects{objectIdx}{2}, objects{objectIdx}{3}, objects{objectIdx}{4}, objects{objectIdx}{5}];
        if isKey(frameObjects, frameIdx)
            frameObjects(frameIdx) = [frameObjects(frameIdx); bbox];
        else
            frameObjects(frameIdx) = bbox;
        end

        printProgressBar(i, numel(objects), startTime);  % Call to function that prints the progress bar
    end

    fprintf('Drawing bounding boxes on frames...\n');
    startTime = tic;
    for frameIdx = 1:numel(imageSequence)
        if isKey(frameObjects, frameIdx)
            img = imageSequence{frameIdx};
            bboxes = frameObjects(frameIdx);
            imgWithBBoxes = insertShape(img, 'Rectangle', bboxes, 'LineWidth', 3, 'Color', 'red');
            imwrite(imgWithBBoxes, fullfile(outputFolder, sprintf('frame_%06d.png', frameIdx)));
        else
            imwrite(imageSequence{frameIdx}, fullfile(outputFolder, sprintf('frame_%06d.png', frameIdx)));
        end
        printProgressBar(frameIdx, numel(imageSequence), startTime);  % Call to function that prints the progress bar
    end
end


function combinedMasks = combineMasks(mask1, mask2) 
    fprintf('Combining masks...\n');
    startTime = tic;
    numFrames = numel(mask1);
    combinedMasks = cell(1, numFrames);
    for frameIdx = 1:numFrames
        if ~isempty(mask1{frameIdx}) && ~isempty(mask2{frameIdx})
            combinedMasks{frameIdx} = mask1{frameIdx} & mask2{frameIdx};
        else 
            error('Masks frame counts do not match.')
        end
        printProgressBar(frameIdx, numFrames, startTime);  % Call to function that prints the progress bar
    end
end

function saveObjectsToTxt(objects, filename)
    fprintf('Saving objects to text file...\n');
    startTime = tic;
    fileID = fopen(filename, 'w');
    for objectIdx = 1:numel(objects)
        fprintf(fileID, '%d, -1, %d, %d, %d, %d, 1, -1, -1, -1\n', objects{objectIdx}{1}, objects{objectIdx}{2}, objects{objectIdx}{3}, objects{objectIdx}{4}, objects{objectIdx}{5});
        printProgressBar(objectIdx, numel(objects), startTime);  % Call to function that prints the progress bar
    end
    fclose(fileID);
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