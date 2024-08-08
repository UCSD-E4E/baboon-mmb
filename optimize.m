function optimize(varargin)
% Optimize function entry point. Parses inputs, configures options,
% performs optimization, and handles results.

% Read configuration file
config = readConfigFile('config.json');

% Convert user-defined parameters
params = convertUserParams(config);

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

% Use configuration values
lb = config.lb;
ub = config.ub;
mu = config.mu;
std = config.std;
intIndices = config.intIndices;

% Adjust upper bounds based on image properties
ub(3) = min(ub(3), frameArea);
ub(4) = min(ub(4), frameArea);
ub(5) = min(ub(5), maxDimension);
ub(6) = min(ub(6), maxDimension);
ub(7) = min(ub(7), frameCount / params.FrameRate);
ub(8) = min(ub(8), maxDimension);
ub(10) = min(ub(10), frameCount - 1);
ub(11) = min(ub(11), frameDiagonal);
ub(12) = min(ub(12), frameCount - 1);

% Conditionally load a saved state or initialize optimization options
options = configureOptions(params, mu, std, lb, ub, intIndices);

% Perform the optimization
[solution, fval, exitFlag, output] = performOptimization(params, options, lb, ub, intIndices);

% Save results and plot the Pareto front
saveOptimizationResults(solution, fval, exitFlag, output);
plotParetoFront(fval);
end

function config = readConfigFile(filename)
% Read and parse the JSON configuration file
fid = fopen(filename, 'r');
if fid == -1
    error('Cannot open configuration file: %s', filename);
end
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
config = jsondecode(str);

% Replace 'Inf' strings with actual Inf values
fields = {'lb', 'ub', 'mu', 'std'};
for i = 1:length(fields)
    field = fields{i};
    config.(field) = cellfun(@(x) str2double(x), config.(field));
    config.(field)(isinf(config.(field)) & config.(field) < 0) = -Inf;
    config.(field)(isinf(config.(field)) & config.(field) > 0) = Inf;
end
end

function params = convertUserParams(config)
% Convert user-defined parameters to appropriate data types
params = struct();
params.InputPath = config.InputPath;
params.GroundTruthPath = config.GroundTruthPath;
params.FrameRate = str2double(config.FrameRate);
params.PopulationSize = str2double(config.PopulationSize);
params.MaxGenerations = str2double(config.MaxGenerations);
params.FunctionTolerance = str2double(config.FunctionTolerance);
params.MaxStallGenerations = str2double(config.MaxStallGenerations);
params.UseParallel = strcmpi(config.UseParallel, 'true');
params.ParetoFraction = str2double(config.ParetoFraction);
params.Display = config.Display;
params.Continue = strcmpi(config.Continue, 'true');
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
            % Generate normally distributed random numbers
            individual = mu + std .* randn(1, numVariables);
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

function [x, fval, exitFlag, output] = performOptimization(params, options, lb, ub, intIndices)
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

for frame = uniqueFrames
    gtObjects = groundTruthData([groundTruthData.frameNumber] == frame);
    detectedObjects = detectedData([detectedData.frameNumber] == frame);
    numGt = length(gtObjects);
    numDet = length(detectedObjects);
    
    % Initialize cost matrix
    cost_matrix = zeros(numGt, numDet);

    % Calcuate IoU for each detection and ground truth pair
    for i = 1:numDet
        for j = 1:numGt
            detBox = [detectedObjects(i).x, detectedObjects(i).y, detectedObjects(i).width, detectedObjects(i).height];
            gtBox = [gtObjects(j).x, gtObjects(j).y, gtObjects(j).width, gtObjects(j).height];

            xD = max([detBox(1), gtBox(1)]);
            yD = max([detBox(2), gtBox(2)]);
            xG = min([detBox(1) + detBox(3), gtBox(1) + gtBox(3)]);
            yG = min([detBox(2) + detBox(4), gtBox(2) + gtBox(4)]);

            % Calculate intersection area
            interArea = max(0, xG - xD) * max(0, yG - yD);

            % Calculate areas of each box
            boxAArea = detBox(3) * detBox(4);
            boxBArea = gtBox(3) * gtBox(4);

            % Compute union area
            unionArea = boxAArea + boxBArea - interArea;

            % Compute IoU
            if unionArea > 0
                iou = interArea / unionArea;
            else
                iou = 0;
            end
            cost_matrix(i, j) = iou;
        end
    end

    iou_threshold = 0.0;

    assignments = matchpairs(-cost_matrix, iou_threshold);

    % Count true positives
    for k = 1:size(assignments, 1)
        if cost_matrix(assignments(k, 1), assignments(k, 2)) > iou_threshold
            TP = TP + 1;
        else
            FP = FP + 1;
            FN = FN + 1;
        end
    end

    % Count false positive and false negative
    FP = FP + (numDet - size(assignments, 1));
    FN = FN + (numGt - size(assignments, 1));
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