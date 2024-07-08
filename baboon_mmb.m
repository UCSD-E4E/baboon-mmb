function objects = baboon_mmb(varargin)
    p = inputParser;

    addParameter(p, 'K', 4, @(x) x >= 0 && x <= 8);
    addParameter(p, 'CONNECTIVITY', 8, @(x) any(x == [4, 8]));
    addParameter(p, 'AREA_MIN', 5, @(x) x >= 0 && x <= 100);
    addParameter(p, 'AREA_MAX', 80, @(x) x >= 0 && x <= 100);
    addParameter(p, 'ASPECT_RATIO_MIN', 1, @(x) x >= 0 && x <= 10);
    addParameter(p, 'ASPECT_RATIO_MAX', 6, @(x) x >= 0 && x <= 10);
    addParameter(p, 'L', 4, @(x) x >= 1 && x <= 10);
    addParameter(p, 'KERNEL', 3, @(x) x >= 1  && x <= 11);
    addParameter(p, 'BITWISE_OR', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'PIPELINE_LENGTH', 5, @(x) x >= 1 && x <= 10);
    addParameter(p, 'PIPELINE_SIZE', 7, @(x) x >= 3 && x <= 11);
    addParameter(p, 'H', 3, @(x) x >= 1 && x <= 10);
    addParameter(p, 'MAX_NITER_PARAM', 10, @(x) x >= 1 && x <= 20);
    addParameter(p, 'GAMMA1_PARAM', 8, @(x) x >= 0 && x <= 1);
    addParameter(p, 'GAMMA2_PARAM', 8, @(x) x >= 0 && x <= 1);
    addParameter(p, 'FRAME_RATE', 10, @(x) x >= 1);
    addParameter(p, 'IMAGE_SEQUENCE', '', @ischar);

    parse(p, varargin{:});
    args = p.Results;
   
    % Define an empty objects structure
    emptyObjects = struct('frameNumber', {}, 'id', {}, 'x', {}, 'y', {}, 'width', {}, 'height', {});

    % Validate parameters
    if args.AREA_MIN > args.AREA_MAX
        objects = emptyObjects;
        return;
    end

    if args.ASPECT_RATIO_MIN > args.ASPECT_RATIO_MAX
        objects = emptyObjects;
        return;
    end

    if args.H > args.PIPELINE_LENGTH
        objects = emptyObjects;
        return;
    end

    if args.GAMMA1_PARAM > args.GAMMA2_PARAM
        objects = emptyObjects;
        return;
    end

    imageSequence = loadImageSequence(args.IMAGE_SEQUENCE);
    grayFrames = cellfun(@(x) rgb2gray(x), imageSequence, 'UniformOutput', false);

    amfdMasks = amfd(args.K, args.CONNECTIVITY, args.AREA_MIN, args.AREA_MAX, args.ASPECT_RATIO_MIN, args.ASPECT_RATIO_MAX, args.KERNEL, grayFrames);
    % saveMasks(amfdMasks, 'output/amfd');
    % save('output/amfdMasks.mat', 'amfdMasks');

    if ~any(cellfun(@(x) any(x(:)), amfdMasks)) && ~args.BITWISE_OR
        objects = emptyObjects;
        return;
    end

    lrmcMasks = lrmc(args.L, args.KERNEL, args.MAX_NITER_PARAM, args.GAMMA1_PARAM, args.GAMMA2_PARAM, args.FRAME_RATE, grayFrames);
    % saveMasks(lrmcMasks, 'output/lrmc');
    % save('output/lrmcMasks.mat', 'lrmcMasks');
    
    if ~any(cellfun(@(x) any(x(:)), lrmcMasks)) && ~args.BITWISE_OR
        objects = emptyObjects;
        return;
    end

    % load('output/amfdMasks.mat', 'amfdMasks');
    % load('output/lrmcMasks.mat', 'lrmcMasks');
    combinedMasks = combineMasks(amfdMasks, lrmcMasks, args.BITWISE_OR);
    % saveMasks(combinedMasks, 'output/combinedMasks');
    % save('output/combinedMasks.mat', 'combinedMasks');

    if ~any(cellfun(@(x) any(x(:)), combinedMasks))
        objects = emptyObjects;
        return;
    end
    
    % load('output/combinedMasks.mat', 'combinedMasks');
    objects = pf(args.PIPELINE_LENGTH, args.PIPELINE_SIZE, args.H, combinedMasks);
    % save('output/objects.mat', 'objects');
    % load('output/objects.mat', 'objects');

    % saveObjectsToTxt(objects, 'output/objects.txt');

    % drawBoundingBoxesOnFrames(imageSequence, objects, 'output/frames');
end 

function imageSequence = loadImageSequence(imagePath)
    % fprintf('Loading image sequence...\n');
    if ~isempty(imagePath)
        files = dir(fullfile(imagePath, '*.jpg'));
        [~, idx] = sort({files.name});
        files = files(idx);
        imageSequence = cell(1, numel(files));
        for fileIdx = 1:numel(files)
            imageSequence{fileIdx} = imread(fullfile(imagePath, files(fileIdx).name));
        end
    else
        error('Image sequence folder is not specified or does not exist.');
    end
end

function saveMasks(output, outputDir)
    fprintf('Saving masks...\n');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    for frameIdx = 1:numel(output)
        if ~isempty(output{frameIdx})
            filename = fullfile(outputDir, sprintf('%06d.png', frameIdx));
            imwrite(output{frameIdx}, filename, 'png');
        end
    end
end

function combinedMasks = combineMasks(mask1, mask2, bitwise_or) 
    % fprintf('Combining masks...\n');
    numFrames = numel(mask1);
    combinedMasks = cell(1, numFrames);
    for frameIdx = 1:numFrames
        if ~isempty(mask1{frameIdx}) && ~isempty(mask2{frameIdx})
            if bitwise_or
                combinedMasks{frameIdx} = mask1{frameIdx} | mask2{frameIdx}; % Bitwise OR
            else
                combinedMasks{frameIdx} = mask1{frameIdx} & mask2{frameIdx}; % Bitwise AND
            end
        else
            error('Masks frame counts do not match.');
        end
    end
end

function saveObjectsToTxt(objects, filename)
    fprintf('Saving objects to text file...\n');
    fileID = fopen(filename, 'w');
    for objectIdx = 1:numel(objects)
        obj = objects(objectIdx);
        fprintf(fileID, '%d, %d, %d, %d, %d, %d, 1, -1, -1, -1\n', obj.frameNumber, obj.id, obj.x, obj.y, obj.width, obj.height);
    end
    fclose(fileID);
end

function drawBoundingBoxesOnFrames(imageSequence, objects, outputFolder)
    fprintf('Sorting bounding boxes...\n');
    
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    % Use a single color for all bounding boxes
    boxColor = uint8([0, 255, 0]); % Red color

    % Group objects by frames
    frameObjects = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    
    for objectIdx = 1:numel(objects)
        frameIdx = objects(objectIdx).frameNumber;
        bbox = [objects(objectIdx).x, objects(objectIdx).y, objects(objectIdx).width, objects(objectIdx).height];
        if isKey(frameObjects, frameIdx)
            frameObjects(frameIdx) = [frameObjects(frameIdx); bbox];
        else
            frameObjects(frameIdx) = bbox;
        end
    end
    
    fprintf('Drawing bounding boxes...\n');
    for frameIdx = 1:numel(imageSequence)
        img = imageSequence{frameIdx};
        if isKey(frameObjects, frameIdx)
            bboxes = frameObjects(frameIdx);
            
            % Draw bounding boxes
            for j = 1:size(bboxes, 1)
                img = insertShapeManual(img, 'rectangle', bboxes(j, :), boxColor, 3);
            end
            
            imwrite(img, fullfile(outputFolder, sprintf('frame_%06d.png', frameIdx)));
        else
            imwrite(img, fullfile(outputFolder, sprintf('frame_%06d.png', frameIdx)));
        end
    end
end

function img = insertShapeManual(img, shapeType, position, color, lineWidth)
    % Manually insert shapes into an image
    switch shapeType
        case 'rectangle'
            x1 = position(1);
            y1 = position(2);
            x2 = x1 + position(3);
            y2 = y1 + position(4);
            img = insertLineManual(img, [x1, y1, x2, y1], color, lineWidth); % Top
            img = insertLineManual(img, [x1, y1, x1, y2], color, lineWidth); % Left
            img = insertLineManual(img, [x2, y1, x2, y2], color, lineWidth); % Right
            img = insertLineManual(img, [x1, y2, x2, y2], color, lineWidth); % Bottom
        case 'line'
            img = insertLineManual(img, position, color, lineWidth);
    end
end

function img = insertLineManual(img, position, color, lineWidth)
    % Manually insert a line into an image
    x1 = position(1);
    y1 = position(2);
    x2 = position(3);
    y2 = position(4);
    img = insertShape(img, 'Line', [x1, y1, x2, y2], 'Color', color, 'LineWidth', lineWidth);
end