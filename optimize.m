function optimize(varargin)
% Set up the input parser and define parameter validations
parser = setupInputParser();
parse(parser, varargin{:});
results = convertParams(parser.Results);

% Conditionally load a saved state or initialize optimization options
options = configureOptions(results);

% Perform the optimization
[solution, fval, exitFlag, output] = performOptimization(params, options);

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
            results.(fnames{i}) = str2double(parsedResults.(fnames{i}));
    end
end
end

function options = configureOptions(params)
% Configure optimization options, optionally continuing from a saved state
stateFile = 'output/gamultiobj_state.mat';
if params.Continue && isfile(stateFile)
    load(stateFile, 'state', 'options');
    options.InitialPopulationMatrix = state.Population;
    options.MaxGenerations = params.MaxGenerations - state.Generation;
    fprintf('Continuing from saved state...\n');
else
    options = optimoptions('gamultiobj', ...
        'PopulationSize', params.PopulationSize, ...
        'MaxGenerations', params.MaxGenerations, ...
        'FunctionTolerance', params.FunctionTolerance, ...
        'MaxStallGenerations', params.MaxStallGenerations, ...
        'UseParallel', params.UseParallel, ...
        'ParetoFraction', params.ParetoFraction, ...
        'Display', params.Display, ...
        'OutputFcn', @saveCheckpoint);
end
end

function [x, fval, exitFlag, output] = performOptimization(params, options)
% Perform the optimization using specified parameters and options
groundTruthData = loadGroundTruth(params.GroundTruthPath);
[height, width] = getImageDimensions(params.InputPath);
frameArea = height * width;
frameCount = numel(dir(fullfile(params.InputPath, '*.jpg')));
frameDiagonal = sqrt(width^2 + height^2);
maxDimension = max(height, width);

FitnessFunction = @(params) evaluateParams(params, groundTruthData, height, width, frameCount, frameArea, frameDiagonal);

% Define bounds and integer constraints for optimization variables
numberOfVariables = 15;
lb = [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0];
ub = [Inf, 2, frameArea, frameArea, maxDimension, maxDimension, ...
    frameCount / params.FrameRate, maxDimension, 2, frameCount - 1, ...
    maxDimension, frameCount - 1, Inf, 1, 1];
intIndices = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13];

% Perform multi-objective optimization
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