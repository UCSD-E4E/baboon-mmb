function optimize(varargin)
% Optimize function entry point. Parses inputs, configures options,
% performs optimization, and handles results.

% Set up the input parser and define parameter validations
parser = setupInputParser();
parse(parser, varargin{:});
params = convertParams(parser.Results);

% Get image dimensions from the first image in the input path
firstImageFile = dir(fullfile(params.InputPath, '*.jpg'));
if isempty(firstImageFile)
    error('No images found in the input path: %s', params.InputPath);
end

try
    firstImage = imread(fullfile(params.InputPath, firstImageFile(1).name));
catch
    error('Failed to read the first image in the input path: %s', fullfile(params.InputPath, firstImageFile(1).name));
end

[height, width, ~] = size(firstImage);
frameArea = height * width;
frameCount = numel(dir(fullfile(params.InputPath, '*.jpg')));
frameDiagonal = sqrt(width^2 + height^2);
maxDimension = max(height, width);

lb = [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0];
ub = [Inf, 2, frameArea, frameArea, maxDimension, maxDimension, ...
    frameCount / params.FrameRate, maxDimension, 2, frameCount - 1, ...
    frameDiagonal, frameCount - 1, Inf, 1, 1];
mu = [4, 2, 5, 80, 1, 6, 4, 3, 1, 5, 7, 3, 10, 0.3, 0.8];
std = min(min([
    1, ...
    0.5, ...
    (ub(3) - lb(3)) / 4, ...
    (ub(4) - lb(4)) / 4, ...
    (ub(5) - lb(5)) / 4, ...
    (ub(6) - lb(6)) / 4, ...
    (ub(7) - lb(7)) / 4, ...
    1, ...
    0.5, ...
    (ub(10) - lb(10)) / 4, ...
    (ub(11) - lb(11)) / 4, ...
    (ub(12) - lb(12)) / 4, ...
    (ub(13) - lb(13)) / 4, ...
    0.1, ...
    0.1], ...
    abs(mu - lb)), abs(mu - ub));
intIndices = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13];

% Conditionally load a saved state or initialize optimization options
options = configureOptions(params, mu, std, lb, ub, intIndices);

% Perform the optimization
[solution, fval, exitFlag, output] = performOptimization(params, options, lb, ub);

% Save results and plot the Pareto front
saveOptimizationResults(solution, fval, exitFlag, output);
plotParetoFront(fval);
end

function parser = setupInputParser()
% Setup input parser with parameter validations
parser = inputParser;

validPath = @(x) ischar(x) && isfolder(x);
validFile = @(x) ischar(x) && isfile(x);
validNumericStr = @(x) ischar(x) && ~isnan(str2double(x));
validBooleanStr = @(x) ischar(x) && any(strcmpi(x, {'true', 'false', '0', '1'}));
validOptType = @(x) ischar(x) && any(str2double(x) == [1, 2, 3, 4]);

addParameter(parser, 'InputPath', 'input/viso_video_1', validPath);
addParameter(parser, 'GroundTruthPath', 'input/viso_video_1_gt.txt', validFile);
addParameter(parser, 'FrameRate', '10', validNumericStr);
addParameter(parser, 'PopulationSize', '1000', validNumericStr);
addParameter(parser, 'MaxGenerations', '1e9', validNumericStr);
addParameter(parser, 'FunctionTolerance', '1e-10', validNumericStr);
addParameter(parser, 'MaxStallGenerations', '1e6', validNumericStr);
addParameter(parser, 'UseParallel', 'true', validBooleanStr);
addParameter(parser, 'ParetoFraction', '0.7', validNumericStr);
addParameter(parser, 'Display', 'iter', @ischar);
addParameter(parser, 'Continue', 'false', validBooleanStr);
addParameter(parser, 'OptimizationType', '2', validOptType);
end

function results = convertParams(parsedResults)
% Convert parameters to their appropriate data types
fnames = fieldnames(parsedResults);
results = struct();
for i = 1:length(fnames)
    switch fnames{i}
        case {'InputPath', 'GroundTruthPath', 'Display'}
            results.(fnames{i}) = parsedResults.(fnames{i});
        case {'UseParallel', 'Continue'}
            results.(fnames{i}) = strcmpi(parsedResults.(fnames{i}), 'true') || str2double(parsedResults.(fnames{i})) == 1;
        otherwise
            value = str2double(parsedResults.(fnames{i}));
            if isnan(value)
                error('Invalid value for parameter %s: %s', fnames{i}, parsedResults.(fnames{i}));
            end
            results.(fnames{i}) = value;
    end
end
end

function options = configureOptions(params, mu, std, lb, ub, intIndices)
% Configure optimization options, optionally continuing from a saved state
stateFile = 'output/gamultiobj_state.mat';
if params.Continue && isfile(stateFile)
    load(stateFile, 'state', 'options');
    options.InitialPopulationMatrix = state.Population;
    options.MaxGenerations = params.MaxGenerations - state.Generation;
    fprintf('Continuing from saved state...\n');
else
    % Generate initial population using mu and std
    populationSize = params.PopulationSize;
    numVariables = length(mu);
    initialPopulation = zeros(populationSize, numVariables);
    
    for i = 1:populationSize
        valid = false;
        while ~valid
            individual = normrnd(mu, std);
            % Ensure the values are within bounds
            if all(individual >= lb) && all(individual <= ub)
                % Ensure integer constraints
                individual(intIndices) = round(individual(intIndices));
                % Check constraints
                if individual(3) <= individual(4) && ...  % AREA_MIN <= AREA_MAX
                        individual(5) <= individual(6) && ...  % ASPECT_RATIO_MIN <= ASPECT_RATIO_MAX
                        individual(12) <= individual(10) && ... % H <= PIPELINE_LENGTH
                        individual(14) <= individual(15)        % GAMMA1_PARAM <= GAMMA2_PARAM
                    valid = true;
                end
            end
        end
        initialPopulation(i, :) = individual;
    end
    
    options = optimoptions('gamultiobj', ...
        'PopulationSize', params.PopulationSize, ...
        'MaxGenerations', params.MaxGenerations, ...
        'FunctionTolerance', params.FunctionTolerance, ...
        'MaxStallGenerations', params.MaxStallGenerations, ...
        'UseParallel', params.UseParallel, ...
        'ParetoFraction', params.ParetoFraction, ...
        'Display', params.Display, ...
        'InitialPopulationMatrix', initialPopulation, ...
        'OutputFcn', @saveCheckpoint);
end
end

function [x, fval, exitFlag, output] = performOptimization(params, options, lb, ub)
% Load and process ground truth data
try
    groundTruthFile = load(params.GroundTruthPath);
catch
    error('Failed to load ground truth file: %s', params.GroundTruthPath);
end
numEntries = size(groundTruthFile, 1);
template = struct('frameNumber', [], 'id', [], 'x', [], 'y', [], 'width', [], 'height', [], 'cx', [], 'cy', []);
groundTruthData = repmat(template, numEntries, 1);
for i = 1:numEntries
    groundTruthData(i).frameNumber = groundTruthFile(i, 1);
    groundTruthData(i).id = groundTruthFile(i, 2);
    groundTruthData(i).x = groundTruthFile(i, 3);
    groundTruthData(i).y = groundTruthFile(i, 4);
    groundTruthData(i).width = groundTruthFile(i, 5);
    groundTruthData(i).height = groundTruthFile(i, 6);
    groundTruthData(i).cx = groundTruthFile(i, 3) + groundTruthFile(i, 5) / 2;
    groundTruthData(i).cy = groundTruthFile(i, 4) + groundTruthFile(i, 6) / 2;
end

FitnessFunction = @(optParams) evaluateParams(optParams, params, groundTruthData);

% Perform multi-objective optimization
numberOfVariables = length(lb);
[x, fval, exitFlag, output] = gamultiobj(FitnessFunction, numberOfVariables, [], [], [], [], lb, ub, @constraintFunction, intIndices, options);

    function [c, ceq] = constraintFunction(x)
        % Define nonlinear inequality and equality constraints
        c = [
            x(3) - x(4);  % AREA_MIN <= AREA_MAX
            x(5) - x(6);  % ASPECT_RATIO_MIN <= ASPECT_RATIO_MAX
            x(12) - x(10); % H <= PIPELINE_LENGTH
            x(14) - x(15); % GAMMA1_PARAM <= GAMMA2_PARAM
            ];
        
        % Nonlinear equality constraints (ceq = 0)
        ceq = [];
    end
end

function [precision, recall] = evaluateParams(optParams, userParams, groundTruthData)
fprintf('Running parameters: %s\n', sprintf('%.4f ', optParams));

% Map the auxiliary variables
connectivityOptions = [4, 8];
connectivityValue = connectivityOptions(optParams(2));
bitwiseOrOptions = [false, true];
bitwiseOrValue = bitwiseOrOptions(optParams(9));

% Initialize detection and set default values for counts
detectedData = baboon_mmb('K', optParams(1), 'CONNECTIVITY', connectivityValue, ...
    'AREA_MIN', optParams(3), 'AREA_MAX', optParams(4), ...
    'ASPECT_RATIO_MIN', optParams(5), 'ASPECT_RATIO_MAX', optParams(6), ...
    'L', optParams(7), 'KERNEL', optParams(8), 'BITWISE_OR', bitwiseOrValue, ...
    'PIPELINE_LENGTH', optParams(10), 'PIPELINE_SIZE', optParams(11), ...
    'H', optParams(12), 'MAX_NITER_PARAM', optParams(13), ...
    'GAMMA1_PARAM', optParams(14), 'GAMMA2_PARAM', optParams(15), ...
    'FRAME_RATE', userParams.FrameRate, 'IMAGE_SEQUENCE', userParams.InputPath, 'DEBUG', false);

TP = 0; FP = 0; FN = 0;

% Ensure detectedData is not empty
if isempty(detectedData)
    detectedData = struct('frameNumber', [], 'id', [], 'x', [], 'y', [], 'width', [], 'height', [], 'cx', [], 'cy', []);
end

% Analyze each unique frame
uniqueFrames = unique([groundTruthData.frameNumber, [detectedData.frameNumber]]);
largeCost = 1e6;

for frame = uniqueFrames
    gtObjects = groundTruthData([groundTruthData.frameNumber] == frame);
    detectedObjects = detectedData([detectedData.frameNumber] == frame);
    numGt = length(gtObjects);
    numDet = length(detectedObjects);
    
    switch userParams.OptimizationType
        case 1
            matchedDetections = false(numDet, 1);
            matchedGroundTruth = false(numGt, 1);
            
            for i = 1:numGt
                bbGt = [gtObjects(i).x, gtObjects(i).y, gtObjects(i).width, gtObjects(i).height];
                
                for j = 1:numDet
                    bbDet = [detectedObjects(j).x, detectedObjects(j).y, detectedObjects(j).width, detectedObjects(j).height];
                    overlapRatio = bboxOverlapRatio(bbGt, bbDet);
                    if overlapRatio > 0
                        matchedDetections(j) = true;
                        matchedGroundTruth(i) = true;
                    end
                end
            end
            
            TP = TP + sum(matchedGroundTruth);
            FP = FP + sum(~matchedDetections);
            FN = FN + sum(~matchedGroundTruth);
        case 2
            costMatrix = largeCost * ones(numGt, numDet);
            
            for i = 1:numGt
                for j = 1:numDet
                    bbGt = [gtObjects(i).x, gtObjects(i).y, gtObjects(i).width, gtObjects(i).height];
                    bbDet = [detectedObjects(j).x, detectedObjects(j).y, detectedObjects(j).width, detectedObjects(j).height];
                    overlapRatio = bboxOverlapRatio(bbGt, bbDet);
                    if overlapRatio > 0
                        costMatrix(i, j) = 1 - overlapRatio;
                    end
                end
            end
            
            [assignments, unassignedRows, unassignedCols] = assignDetectionsToTracks(costMatrix, largeCost - 1);
            TP = TP + size(assignments, 1);
            FP = FP + length(unassignedCols);
            FN = FN + length(unassignedRows);
        case 3
            matchedGt = false(1, numGt);
            for j = 1:numDet
                bbDet = [detectedObjects(j).x, detectedObjects(j).y, detectedObjects(j).width, detectedObjects(j).height];
                maxOverlap = 0;
                bestMatchedIdx = 0;
                for i = 1:numGt
                    if ~matchedGt(i)
                        bbGt = [gtObjects(i).x, gtObjects(i).y, gtObjects(i).width, gtObjects(i).height];
                        overlapRatio = bboxOverlapRatio(bbGt, bbDet);
                        if overlapRatio > maxOverlap
                            maxOverlap = overlapRatio;
                            bestMatchedIdx = i;
                        end
                        
                    end
                end
                if maxOverlap > 0
                    TP = TP + 1;
                    matchedGt(bestMatchedIdx) = true;
                else
                    FP = FP + 1;
                end
            end
            
            FN = FN + sum(~matchedGt);
        case 4
            matchedGt = false(1, numGt);
            for j = 1:numDet
                bbDet = [detectedObjects(j).x, detectedObjects(j).y, detectedObjects(j).width, detectedObjects(j).height];
                perfectMatch = false;
                for i = 1:numGt
                    if ~matchedGt(i)
                        bbGt = [gtObjects(i).x, gtObjects(i).y, gtObjects(i).width, gtObjects(i).height];
                        if isequal(bbGt, bbDet)
                            TP = TP + 1;
                            matchedGt(i) = true;
                            perfectMatch = true;
                            break;
                        end
                    end
                end
                if ~perfectMatch
                    FP = FP + 1;
                end
            end
            FN = FN + sum(~matchedGt);
    end
end

% Calculate precision, recall, and F1-score
if (TP + FP) == 0
    precision = 0;
else
    precision = TP / (TP + FP);
end

if (TP + FN) == 0
    recall = 0;
else
    recall = TP / (TP + FN);
end

if (precision + recall) == 0
    f1Score = 0;
else
    f1Score = 2 * (precision * recall) / (precision + recall);
end

% Log results
fprintf('Precision: %.4f Recall: %.4f F1: %.4f\n', precision, recall, f1Score);
outputDir = 'output/';
if ~isfolder(outputDir)
    mkdir(outputDir);
end
uniqueID = tempname(outputDir); % Generates a unique file name
[~, uniqueFileName, ~] = fileparts(uniqueID); % Extracts the unique part of the file name
resultsFile = fullfile(outputDir, [uniqueFileName, '.txt']); % Adds .txt extension
paramStr = sprintf('%.4f ', optParams);
fileID = fopen(resultsFile, 'a');
if fileID == -1
    error('Failed to open results file: %s', resultsFile);
end
fprintf(fileID, '%s Precision: %.4f Recall: %.4f F1: %.4f\n', paramStr, precision, recall, f1Score);
fclose(fileID);
end

function saveOptimizationResults(x, Fval, exitFlag, Output)
outputDir = 'output/';
if ~isfolder(outputDir)
    mkdir(outputDir);
end
save(fullfile(outputDir, 'final_pareto_solutions.mat'), 'x', 'Fval', 'exitFlag', 'Output');
end

function plotParetoFront(Fval)
outputDir = 'output/';
if ~isfolder(outputDir)
    mkdir(outputDir);
end
figure;
plot(Fval(:,1), Fval(:,2), 'bo');
xlabel('Precision');
ylabel('Recall');
title('Pareto Front');
saveas(gcf, fullfile(outputDir, 'pareto_front.png'));
end

function [state, options, optchanged] = saveCheckpoint(options, state, flag)
optchanged = false;
if strcmp(flag, 'iter') || strcmp(flag, 'diagnose')
    save('output/gamultiobj_state.mat', 'state', 'options');
end
end