function optimize()
% Optimize function entry point. Parses inputs, configures options,
% performs optimization, and handles results.

% Read configuration from file
config = readConfigFile('optimize_config.txt');

% Get image dimensions from the first image in the input path
firstImageFile = dir(fullfile(config.InputPath, '*.jpg'));
if isempty(firstImageFile)
    error('No images found in the input path: %s', config.InputPath);
end

try
    firstImage = imread(fullfile(config.InputPath, firstImageFile(1).name));
catch
    error('Failed to read the first image in the input path: %s', fullfile(config.InputPath, firstImageFile(1).name));
end

[height, width, ~] = size(firstImage);
frameArea = height * width;
frameCount = numel(dir(fullfile(config.InputPath, '*.jpg')));
frameDiagonal = sqrt(width^2 + height^2);
maxDimension = max(height, width);

% Adjust upper bounds based on frame properties
config.ub(3) = min(config.ub(3), frameArea);
config.ub(4) = min(config.ub(4), frameArea);
config.ub(5) = min(config.ub(5), maxDimension);
config.ub(6) = min(config.ub(6), maxDimension);
config.ub(7) = min(config.ub(7), frameCount / config.FrameRate);
config.ub(8) = min(config.ub(8), maxDimension);
config.ub(10) = min(config.ub(10), frameCount - 1);
config.ub(11) = min(config.ub(11), frameDiagonal);
config.ub(12) = min(config.ub(12), frameCount - 1);

% Conditionally load a saved state or initialize optimization options
options = configureOptions(config);

% Perform the optimization
[solution, fval, exitFlag, output] = performOptimization(config, options);

% Save results and plot the Pareto front
saveOptimizationResults(solution, fval, exitFlag, output);
plotParetoFront(fval);
end

function config = readConfigFile(filename)
% Read configuration from file
config = struct();
fid = fopen(filename, 'r');
if fid == -1
    error('Cannot open configuration file: %s', filename);
end

while ~feof(fid)
    line = fgetl(fid);
    [key, value] = strtok(line, '=');
    key = strtrim(key);
    value = strtrim(value(2:end));
    
    switch key
        case {'lb', 'ub', 'mu', 'std', 'intIndices'}
            % Convert string to numeric array, handling Inf and decimals
            numericArray = str2num(value);
            if isempty(numericArray)
                error('Invalid numeric array for %s: %s', key, value);
            end
            config.(key) = numericArray;
        case {'InputPath', 'GroundTruthPath', 'Display'}
            config.(key) = value;
        case {'UseParallel', 'Continue'}
            config.(key) = strcmpi(value, 'true') || str2double(value) == 1;
        otherwise
            % For all other numeric parameters
            numValue = str2double(value);
            if isnan(numValue)
                error('Invalid numeric value for %s: %s', key, value);
            end
            config.(key) = numValue;
    end
end

fclose(fid);

% Check if all required fields are present
requiredFields = {'lb', 'ub', 'mu', 'std', 'intIndices', 'InputPath', 'GroundTruthPath', ...
    'FrameRate', 'PopulationSize', 'MaxGenerations', 'FunctionTolerance', ...
    'MaxStallGenerations', 'UseParallel', 'ParetoFraction', 'Display', ...
    'Continue', 'OptimizationType'};
for i = 1:length(requiredFields)
    if ~isfield(config, requiredFields{i})
        error('Missing required configuration: %s', requiredFields{i});
    end
end

% Validate configuration
validateConfig(config);
end

function validateConfig(config)
% Validate the configuration
if numel(config.lb) ~= numel(config.ub) || numel(config.lb) ~= numel(config.mu) || numel(config.lb) ~= numel(config.std)
    error('Inconsistent array lengths for lb, ub, mu, and std');
end

if any(config.lb > config.ub)
    error('Lower bounds must be less than or equal to upper bounds');
end

if any(config.mu < config.lb) || any(config.mu > config.ub)
    error('Initial values (mu) must be within bounds');
end

if any(config.std <= 0)
    error('Standard deviations must be positive');
end

if any(config.intIndices < 1) || any(config.intIndices > numel(config.lb))
    error('Invalid intIndices: must be within the range of parameter indices');
end
end

function options = configureOptions(config)
% Configure optimization options, optionally continuing from a saved state
stateFile = 'output/gamultiobj_state.mat';
if config.Continue && isfile(stateFile)
    load(stateFile, 'state', 'options');
    options.InitialPopulationMatrix = state.Population;
    options.MaxGenerations = config.MaxGenerations - state.Generation;
    fprintf('Continuing from saved state...\n');
else
    % Generate initial population using mu and std
    populationSize = config.PopulationSize;
    numVariables = length(config.mu);
    initialPopulation = zeros(populationSize, numVariables);
    
    for i = 1:populationSize
        valid = false;
        while ~valid
            % Generate normally distributed random numbers
            individual = config.mu + config.std .* randn(1, numVariables);
            % Ensure the values are within bounds
            if all(individual >= config.lb) && all(individual <= config.ub)
                % Ensure integer constraints
                individual(config.intIndices) = round(individual(config.intIndices));
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
        'PopulationSize', config.PopulationSize, ...
        'MaxGenerations', config.MaxGenerations, ...
        'FunctionTolerance', config.FunctionTolerance, ...
        'MaxStallGenerations', config.MaxStallGenerations, ...
        'UseParallel', config.UseParallel, ...
        'ParetoFraction', config.ParetoFraction, ...
        'Display', config.Display, ...
        'InitialPopulationMatrix', initialPopulation, ...
        'OutputFcn', @saveCheckpoint);
end
end

function [x, fval, exitFlag, output] = performOptimization(config, options)
% Load and process ground truth data
try
    groundTruthFile = load(config.GroundTruthPath);
catch
    error('Failed to load ground truth file: %s', config.GroundTruthPath);
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

FitnessFunction = @(optParams) evaluateParams(optParams, config, groundTruthData);

% Perform multi-objective optimization
numberOfVariables = length(config.lb);
[x, fval, exitFlag, output] = gamultiobj(FitnessFunction, numberOfVariables, [], [], [], [], config.lb, config.ub, @constraintFunction, config.intIndices, options);

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

function [precision, recall] = evaluateParams(optParams, config, groundTruthData)
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
    'FRAME_RATE', config.FrameRate, 'IMAGE_SEQUENCE', config.InputPath, 'DEBUG', false);

TP = 0; FP = 0; FN = 0;

% Ensure detectedData is not empty
if isempty(detectedData)
    detectedData = struct('frameNumber', [], 'id', [], 'x', [], 'y', [], 'width', [], 'height', [], 'cx', [], 'cy', []);
end

% Analyze each unique frame
uniqueFrames = unique([groundTruthData.frameNumber, [detectedData.frameNumber]]);

for frame = uniqueFrames
    gtObjects = groundTruthData([groundTruthData.frameNumber] == frame);
    detectedObjects = detectedData([detectedData.frameNumber] == frame);
    numGt = length(gtObjects);
    numDet = length(detectedObjects);
    
    switch config.OptimizationType
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
            costMatrix = zeros(numGt, numDet);
            
            for i = 1:numGt
                bbGt = [gtObjects(i).x, gtObjects(i).y, gtObjects(i).width, gtObjects(i).height];
                bbGt = [bbGt(1), bbGt(2), bbGt(1)+bbGt(3), bbGt(2)+bbGt(4)]; % Convert to [x1,y1,x2,y2] format
                
                for j = 1:numDet
                    bbDet = [detectedObjects(j).x, detectedObjects(j).y, detectedObjects(j).width, detectedObjects(j).height];
                    bbDet = [bbDet(1), bbDet(2), bbDet(1)+bbDet(3), bbDet(2)+bbDet(4)]; % Convert to [x1,y1,x2,y2] format
                    
                    % Calculate IoU
                    xx1 = max(bbGt(1), bbDet(1));
                    yy1 = max(bbGt(2), bbDet(2));
                    xx2 = min(bbGt(3), bbDet(3));
                    yy2 = min(bbGt(4), bbDet(4));
                    
                    w = max(0, xx2 - xx1);
                    h = max(0, yy2 - yy1);
                    
                    wh = w * h;
                    iou = wh / ((bbGt(3)-bbGt(1))*(bbGt(4)-bbGt(2)) + (bbDet(3)-bbDet(1))*(bbDet(4)-bbDet(2)) - wh + 1e-7);
                    
                    costMatrix(i, j) = -iou; 
                end
            end
            
            costOfNonAssignment = 0; 
            [assignments, ~, ~] = assignDetectionsToTracks(costMatrix, costOfNonAssignment);
            
            validAssignments = assignments(costMatrix(sub2ind(size(costMatrix), assignments(:,1), assignments(:,2))) < 0);
            
            TP = size(validAssignments, 1);
            FP = numDet - TP;
            FN = numGt - TP;
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