function optimize(varargin)
% Optimize function entry point. Parses inputs, configures options,
% performs optimization, and handles results.

% Parse input arguments for seed
p = inputParser;
addOptional(p, 'seed', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});

% Set random seed
if isempty(p.Results.seed)
    % Generate a seed based on current time if not provided
    rng('shuffle');
    seed = rng().Seed;
else
    seed = p.Results.seed;
end
rng(seed);
fprintf('Using random seed: %d\n', seed);

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

% Load and process ground truth data
groundTruthData = loadGroundTruth(params.GroundTruthPath);

% Analyze ground truth data
[~, ~, ~, ~, areaMin, areaMax, aspectRatioMin, aspectRatioMax] = analyzeGroundTruth(groundTruthData);

% Use configuration values
lb = config.lb;
ub = config.ub;
mu = config.mu;
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

% Update mu with inferred values, respecting adjusted bounds
mu(3) = max(lb(3), min(ub(3), areaMin));
mu(4) = max(lb(4), min(ub(4), areaMax));
mu(5) = max(lb(5), min(ub(5), aspectRatioMin));
mu(6) = max(lb(6), min(ub(6), aspectRatioMax));

% Now update the std values
std = config.std;
std(3) = min([abs(mu(3) - lb(3)), abs(ub(3) - mu(3)), abs(mu(4) - mu(3))]);
std(4) = min([abs(mu(4) - lb(4)), abs(ub(4) - mu(4)), abs(mu(4) - mu(3))]);
std(5) = min([abs(mu(5) - lb(5)), abs(ub(5) - mu(5)), abs(mu(6) - mu(5))]);
std(6) = min([abs(mu(6) - lb(6)), abs(ub(6) - mu(6)), abs(mu(6) - mu(5))]);

% Configure optimization options
options = configureOptions(params, mu, std, lb, ub, intIndices);

% Perform the optimization
[solution, ~, ~, ~] = performOptimization(params, options, lb, ub, intIndices, groundTruthData);

% Save the solution to a file
save('output/solution.mat', 'solution', 'seed');
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
end

function options = configureOptions(params, mu, std, lb, ub, intIndices)
% Configure optimization options

% Read and sort existing scores
[existingParams, existingScores] = readExistingScores('output');

% Determine how many existing solutions to use
populationSize = params.PopulationSize;
numExisting = min(size(existingParams, 1), populationSize);

% Initialize the population matrix
initialPopulation = zeros(populationSize, length(mu));

% Fill in existing high-scoring solutions
initialPopulation(1:numExisting, :) = existingParams(1:numExisting, :);

% Generate the rest of the population if needed
for i = (numExisting + 1):populationSize
    valid = false;
    while ~valid
        % Generate normally distributed random numbers
        individual = (mu + std .* randn(length(mu), 1))';
        % Ensure the values are within bounds
        if all(individual >= lb' & individual <= ub')
            % Ensure integer constraints
            individual(intIndices) = round(individual(intIndices));
            % Check constraints
            if individual(3) <= individual(4) && ...  % AREA_MIN <= AREA_MAX
                    individual(5) <= individual(6) && ...  % ASPECT_RATIO_MIN <= ASPECT_RATIO_MAX
                    individual(12) <= individual(10) && ... % H <= PIPELINE_LENGTH
                    individual(14) <= individual(15)  % GAMMA1_PARAM <= GAMMA2_PARAM
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
    'InitialPopulationMatrix', initialPopulation);
end

function [sortedParams, sortedScores] = readExistingScores(outputDir)
    % Read all score files in the output directory
    files = dir(fullfile(outputDir, '*_score.txt'));
    params = [];
    scores = [];
    
    for i = 1:length(files)
        filename = fullfile(outputDir, files(i).name);
        fid = fopen(filename, 'r');
        if fid == -1
            warning('Could not open file: %s', filename);
            continue;
        end
        
        line = fgetl(fid);
        fclose(fid);
        
        % Parse the line
        parts = strsplit(line);
        if length(parts) >= 4
            paramValues = str2double(parts(1:end-3));
            precision = str2double(parts{end-2});
            recall = str2double(parts{end-1});
            f1 = str2double(parts{end});
            
            params = [params; paramValues];
            scores = [scores; f1]; % Using F1 score for ranking
        end
    end
    
    % Sort params by score in descending order
    [sortedScores, sortIndex] = sort(scores, 'descend');
    sortedParams = params(sortIndex, :);
end

function [x, fval, exitFlag, output] = performOptimization(params, options, lb, ub, intIndices, groundTruthData)
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
% Get current date and time
currentDateTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');

fprintf('%s - Running parameters: %s\n', currentDateTime, sprintf('%.4f ', optParams));

% Generate a unique filename for the score file
paramStr = sprintf('%.4f_', optParams);
paramHash = generateHash(paramStr);
scoreFile = fullfile('output', [paramHash, '_score.txt']);

% Check if the score file already exists
if exist(scoreFile, 'file')
    % Read the existing score file
    fileID = fopen(scoreFile, 'r');
    if fileID == -1
        error('Failed to open existing score file: %s', scoreFile);
    end
    scoreData = textscan(fileID, '%*s %f %f %f');
    fclose(fileID);
    
    % Extract precision and recall from the file
    precision = scoreData{1};
    recall = scoreData{2};
    
    fprintf('%s - Using existing scores - Precision: %.4f Recall: %.4f\n', currentDateTime, precision, recall);
    return;
end

% Map the auxiliary variables
connectivityOptions = [4, 8];
connectivityValue = connectivityOptions(optParams(2));
bitwiseOrOptions = [false, true];
bitwiseOrValue = bitwiseOrOptions(optParams(9));

try
    % Initialize detection and set default values for counts
    detectedData = baboon_mmb('K', optParams(1), 'CONNECTIVITY', connectivityValue, ...
        'AREA_MIN', optParams(3), 'AREA_MAX', optParams(4), ...
        'ASPECT_RATIO_MIN', optParams(5), 'ASPECT_RATIO_MAX', optParams(6), ...
        'L', optParams(7), 'KERNEL', optParams(8), 'BITWISE_OR', bitwiseOrValue, ...
        'PIPELINE_LENGTH', optParams(10), 'PIPELINE_SIZE', optParams(11), ...
        'H', optParams(12), 'MAX_NITER_PARAM', optParams(13), ...
        'GAMMA1_PARAM', optParams(14), 'GAMMA2_PARAM', optParams(15), ...
        'FRAME_RATE', userParams.FrameRate, 'IMAGE_SEQUENCE', userParams.InputPath, 'DEBUG', false);
catch e
    % If baboon_mmb crashes, log the error and return a score of 0
    fprintf('%s - Error in baboon_mmb: %s\n', currentDateTime, e.message);
    precision = 0;
    recall = 0;
    
    % Log results
    fprintf('%s - Precision: 0.0000 Recall: 0.0000 F1: 0.0000\n', currentDateTime);
    outputDir = 'output/';
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end
    fileID = fopen(scoreFile, 'a');
    if fileID == -1
        error('Failed to open score file: %s', scoreFile);
    end
    fprintf(fileID, '%s %.4f %.4f %.4f\n', paramStr, precision, recall, 0);
    fclose(fileID);
    
    return;
end

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
    
    % Calculate IoU for each detection and ground truth pair
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
fprintf('%s - Precision: %.4f Recall: %.4f F1: %.4f\n', currentDateTime, precision, recall, f1Score);
outputDir = 'output/';
if ~isfolder(outputDir)
    mkdir(outputDir);
end
fileID = fopen(scoreFile, 'a');
if fileID == -1
    error('Failed to open score file: %s', scoreFile);
end
fprintf(fileID, '%s %.4f %.4f %.4f\n', paramStr, precision, recall, f1Score);
fclose(fileID);
end

function hash = generateHash(inputStr)
% Generate an MD5 hash for the input string
md = java.security.MessageDigest.getInstance('MD5');
md.update(uint8(inputStr));
hash = sprintf('%02x', typecast(md.digest(), 'uint8'));
end

function [areaMu, areaStd, aspectRatioMu, aspectRatioStd, areaMin, areaMax, aspectRatioMin, aspectRatioMax] = analyzeGroundTruth(groundTruthData)
    areas = [];
    aspectRatios = [];
    
    for i = 1:numel(groundTruthData)
        area = groundTruthData(i).width * groundTruthData(i).height;
        aspectRatio = max(groundTruthData(i).width, groundTruthData(i).height) / min(groundTruthData(i).width, groundTruthData(i).height);
        
        areas = [areas, area];
        aspectRatios = [aspectRatios, aspectRatio];
    end
    
    % Calculate statistics
    areaMu = mean(areas);
    areaStd = std(areas);
    aspectRatioMu = mean(aspectRatios);
    aspectRatioStd = std(aspectRatios);
    
    % Calculate min and max values
    areaMin = max(1, floor(min(areas) - 0.5 * areaStd));
    areaMax = ceil(max(areas) + 0.5 * areaStd);
    aspectRatioMin = max(1, floor(min(aspectRatios) - 0.5 * aspectRatioStd));
    aspectRatioMax = ceil(max(aspectRatios) + 0.5 * aspectRatioStd);
end

function groundTruthData = loadGroundTruth(groundTruthPath)
    try
        groundTruthFile = load(groundTruthPath);
    catch
        error('Failed to load ground truth file: %s', groundTruthPath);
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
end