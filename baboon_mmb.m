function objects = baboon_mmb(varargin)
p = inputParser;

% Define parameters with validation functions
addParameter(p, 'K', 4, @(x) isnumeric(x) && x >= 0 && x <= 8);
addParameter(p, 'CONNECTIVITY', 8, @(x) isnumeric(x) && any(x == [4, 8]));
addParameter(p, 'AREA_MIN', 5, @(x) isnumeric(x) && x >= 0 && x <= 100);
addParameter(p, 'AREA_MAX', 80, @(x) isnumeric(x) && x >= 0 && x <= 100);
addParameter(p, 'ASPECT_RATIO_MIN', 1, @(x) isnumeric(x) && x >= 0 && x <= 10);
addParameter(p, 'ASPECT_RATIO_MAX', 6, @(x) isnumeric(x) && x >= 0 && x <= 10);
addParameter(p, 'L', 4, @(x) isnumeric(x) && x >= 1 && x <= 10);
addParameter(p, 'KERNEL', 3, @(x) isnumeric(x) && x >= 1 && x <= 11);
addParameter(p, 'BITWISE_OR', false, @(x) islogical(x));
addParameter(p, 'PIPELINE_LENGTH', 5, @(x) isnumeric(x) && x >= 1 && x <= 10);
addParameter(p, 'PIPELINE_SIZE', 7, @(x) isnumeric(x) && x >= 1 && x <= 11);
addParameter(p, 'H', 3, @(x) isnumeric(x) && x >= 1 && x <= 10);
addParameter(p, 'MAX_NITER_PARAM', 10, @(x) isnumeric(x) && x >= 1 && x <= 20);
addParameter(p, 'GAMMA1_PARAM', 8, @(x) isnumeric(x) && x >= 0 && x <= 1);
addParameter(p, 'GAMMA2_PARAM', 8, @(x) isnumeric(x) && x >= 0 && x <= 1);
addParameter(p, 'FRAME_RATE', 10, @(x) isnumeric(x) && x >= 1);
addParameter(p, 'IMAGE_SEQUENCE', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'DEBUG', true, @(x) islogical(x));

parse(p, varargin{:});
args = p.Results;

% Adjust parameters
args.K = double(args.K);
args.CONNECTIVITY = uint8(args.CONNECTIVITY);
args.AREA_MIN = double(args.AREA_MIN);
args.AREA_MAX = double(args.AREA_MAX);
args.ASPECT_RATIO_MIN = double(args.ASPECT_RATIO_MIN);
args.ASPECT_RATIO_MAX = double(args.ASPECT_RATIO_MAX);
args.L = uint8(args.L);
args.KERNEL = uint8(args.KERNEL);
args.BITWISE_OR = logical(args.BITWISE_OR);
args.PIPELINE_LENGTH = uint8(args.PIPELINE_LENGTH);
args.PIPELINE_SIZE = uint8(args.PIPELINE_SIZE);
args.H = uint8(args.H);
args.MAX_NITER_PARAM = uint8(args.MAX_NITER_PARAM);
args.GAMMA1_PARAM = double(args.GAMMA1_PARAM);
args.GAMMA2_PARAM = double(args.GAMMA2_PARAM);
args.FRAME_RATE = uint8(args.FRAME_RATE);

% Check if /output folder exists
if ~exist('output', 'dir')
    mkdir('output');
end

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

if args.DEBUG
    saveMasks(amfdMasks, 'output/amfd');
end

save('output/amfdMasks.mat', 'amfdMasks');

if ~any(cellfun(@(x) any(x(:)), amfdMasks)) && ~args.BITWISE_OR
    objects = emptyObjects;
    return;
end

lrmcMasks = lrmc(args.L, args.KERNEL, args.MAX_NITER_PARAM, args.GAMMA1_PARAM, args.GAMMA2_PARAM, args.FRAME_RATE, grayFrames);
if args.DEBUG
    saveMasks(lrmcMasks, 'output/lrmc');
end
save('output/lrmcMasks.mat', 'lrmcMasks');
if ~any(cellfun(@(x) any(x(:)), lrmcMasks)) && ~args.BITWISE_OR
    objects = emptyObjects;
    return;
end

combinedMasks = combineMasks(amfdMasks, lrmcMasks, args.BITWISE_OR);
if args.DEBUG
    saveMasks(combinedMasks, 'output/combined');
end
save('output/combinedMasks.mat', 'combinedMasks');

if ~any(cellfun(@(x) any(x(:)), combinedMasks))
    objects = emptyObjects;
    return;
end

objects = pf(args.PIPELINE_LENGTH, args.PIPELINE_SIZE, args.H, combinedMasks);
save('output/objects.mat', 'objects');

if args.DEBUG
    saveObjectsToTxt(objects, 'output/objects.txt');
    drawBoundingBoxesOnFrames(imageSequence, objects, 'output/frames');
end
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