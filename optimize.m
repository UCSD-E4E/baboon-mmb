function optimize(varargin)
% Set up the input parser and define parameter validations
p = setupInputParser();
parse(p, varargin{:});
results = convertParams(p.Results);

% Conditionally load a saved state or initialize optimization options
options = configureOptions(results);

% Perform the optimization
[x, Fval, exitFlag, Output] = performOptimization(results, options);

% Save results and plot the Pareto front
saveOptimizationResults(x, Fval, exitFlag, Output);
plotParetoFront(Fval);
end

function p = setupInputParser()
validPath = @(x) ischar(x) && isfolder(x);
validFile = @(x) ischar(x) && isfile(x);
validNumericStr = @(x) ischar(x) && ~isnan(str2double(x));
validBooleanStr = @(x) ischar(x) && any(strcmpi(x, {'true', 'false', '0', '1'}));
validOptType = @(x) ischar(x) && any(str2double(x) == [1, 2, 3, 4]);

p = inputParser;
addParameter(p, 'InputPath', 'input/viso_video_1', validPath);
addParameter(p, 'GroundTruthPath', 'input/viso_video_1_gt.txt', validFile);
addParameter(p, 'FrameRate', '10', validNumericStr);
addParameter(p, 'PopulationSize', '1000', validNumericStr);
addParameter(p, 'MaxGenerations', '1e9', validNumericStr);
addParameter(p, 'FunctionTolerance', '1e-10', validNumericStr);
addParameter(p, 'MaxStallGenerations', '1e6', validNumericStr);
addParameter(p, 'UseParallel', 'true', validBooleanStr);
addParameter(p, 'ParetoFraction', '0.7', validNumericStr);
addParameter(p, 'Display', 'iter', @ischar);
addParameter(p, 'Continue', 'false', validBooleanStr);
addParameter(p, 'OptimizationType', '2', validOptType);
end

function results = convertParams(pResults)
% Convert all parameters to their appropriate data types
fnames = fieldnames(pResults);
for i = 1:length(fnames)
    if contains(fnames{i}, 'Path') || strcmp(fnames{i}, 'Display')
        results.(fnames{i}) = pResults.(fnames{i});
    elseif strcmp(fnames{i}, 'UseParallel') || strcmp(fnames{i}, 'Continue')
        results.(fnames{i}) = strcmpi(pResults.(fnames{i}), 'true') || str2double(pResults.(fnames{i})) == 1;
    else
        results.(fnames{i}) = str2double(pResults.(fnames{i}));
    end
end
end

function options = configureOptions(results)
stateFile = 'output/gamultiobj_state.mat';
if results.Continue && isfile(stateFile)
    load(stateFile, 'state', 'options');
    options.InitialPopulationMatrix = state.Population;
    options.MaxGenerations = results.MaxGenerations - state.Generation;
    fprintf('Continuing from saved state...\n');
else
    options = optimoptions('gamultiobj', 'PopulationSize', results.PopulationSize, 'MaxGenerations', results.MaxGenerations, 'FunctionTolerance', results.FunctionTolerance, 'MaxStallGenerations', results.MaxStallGenerations, 'UseParallel', results.UseParallel, 'ParetoFraction', results.ParetoFraction, 'Display', results.Display, 'OutputFcn', @saveCheckpoint);
end
end

function [x, Fval, exitFlag, Output] = performOptimization(results, options)
groundTruthFile = load(results.GroundTruthPath);
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

% Load the fisrt image to get dimensions
firstImage = imread(fullfile(results.InputPath, dir(fullfile(results.InputPath, '*.jpg')).name));
[height, width, ~] = size(firstImage);
maxDimension = max(height, width);
frameCount = numel(dir(fullfile(results.InputPath, '*.jpg')));
frameArea = height * width;
frameDiagonal = sqrt(width^2 + height^2);

FitnessFunction = @(params) evaluateParams(params, results, groundTruthData, height, width, maxDimension, frameCount, frameArea, frameDiagonal);

% Define the number of variables and constraints for the optimization
numberOfVariables = 15;

% Define the lower and upper bounds for each parameter
lb = [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0];
ub = [Inf, 2, frameArea, frameArea, maxDimension, maxDimension, ...
    frameCount / results.FrameRate, maxDimension, 2, frameCount - 1, ...
    maxDimension, frameCount - 1, Inf, 1, 1];

% Specify which indices are integers
intIndices = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13];

% Use gamultiobj for multi-objective optimization with constraints
[x, Fval, exitFlag, Output] = gamultiobj(FitnessFunction, numberOfVariables, [], [], [], [], lb, ub, @constraintFunction, intIndices, options);

    function [c, ceq] = constraintFunction(x)
        % Nonlinear inequality constraints (c <= 0)
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

function saveOptimizationResults(x, Fval, exitFlag, Output)
save('output/final_pareto_solutions.mat', 'x', 'Fval', 'exitFlag', 'Output');
end

function plotParetoFront(Fval)
figure;
plot(Fval(:,1), Fval(:,2), 'bo');
xlabel('Precision');
ylabel('Recall');
title('Pareto Front');
saveas(gcf, 'output/pareto_front.png');
end

function [state, options, optchanged] = saveCheckpoint(options, state, flag)
optchanged = false;
if strcmp(flag, 'iter') || strcmp(flag, 'diagnose')
    save('output/gamultiobj_state.mat', 'state', 'options');
end
end

function [precision, recall] = evaluateParams(params, results, groundTruthData)
fprintf('Running parameters: %s\n', sprintf('%.4f ', params));

% Map the auxiliary variables
connectivityOptions = [4, 8];
connectivityValue = connectivityOptions(params(2));
bitwiseOrOptions = [false, true];
bitwiseOrValue = bitwiseOrOptions(params(9));


% Initialize detection and set default values for counts
detectedData = baboon_mmb('K', params(1), 'CONNECTIVITY', connectivityValue, ...
    'AREA_MIN', params(3), 'AREA_MAX', params(4), ...
    'ASPECT_RATIO_MIN', params(5), 'ASPECT_RATIO_MAX', params(6), ...
    'L', params(7), 'KERNEL', params(8), 'BITWISE_OR', bitwiseOrValue, ...
    'PIPELINE_LENGTH', params(10), 'PIPELINE_SIZE', params(11), ...
    'H', params(12), 'MAX_NITER_PARAM', params(13), ...
    'GAMMA1_PARAM', params(14), 'GAMMA2_PARAM', params(15), ...
    'FRAME_RATE', results.FrameRate, 'IMAGE_SEQUENCE', results.InputPath, 'DEBUG', false);


fprintf('Evaluating parameters: %s\n', sprintf('%.4f ', params));
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
    
    switch results.OptimizationType
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
paramStr = sprintf('%.4f ', params);
fileID = fopen(resultsFile, 'a');
fprintf(fileID, '%s Precision: %.4f Recall: %.4f F1: %.4f\n', paramStr, precision, recall, f1Score);
fclose(fileID);
end