function optimize(varargin)
% Parse command line arguments
p = inputParser;
addParameter(p, 'InputPath', 'input/viso_video_1', @ischar);
addParameter(p, 'FrameRate', 10, @isnumeric);
parse(p, varargin{:});

inputPath = p.Results.InputPath;
frameRate = p.Results.FrameRate;

% Load ground truth data function
    function groundTruthData = loadGroundTruth(filename)
        groundTruthFile = load(filename);
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

% Evaluate parameters function
    function [negPrecision, negRecall] = evaluateParams(params)
        detectedData = baboon_mmb('K', params(1), 'CONNECTIVITY', 8, ...
            'AREA_MIN', params(2), 'AREA_MAX', params(3), ...
            'ASPECT_RATIO_MIN', params(4), 'ASPECT_RATIO_MAX', params(5), ...
            'L', params(6), 'KERNEL', 3, 'BITWISE_OR', false, ...
            'PIPELINE_LENGTH', params(7), 'PIPELINE_SIZE', params(8), ...
            'H', params(9), 'MAX_NITER_PARAM', params(10), ...
            'GAMMA1_PARAM', params(11), 'GAMMA2_PARAM', params(12), ...
            'FRAME_RATE', frameRate, 'IMAGE_SEQUENCE', inputPath, 'DEBUG', false);
        
        groundTruthData = loadGroundTruth('input/viso_video_1_gt.txt');
        TP = 0; FP = 0; FN = 0;
        
        if isempty(detectedData)
            detectedData = struct('frameNumber', [], 'id', [], 'x', [], 'y', [], 'width', [], 'height', [], 'cx', [], 'cy', []);
        end
        
        uniqueFrames = unique([groundTruthData.frameNumber, [detectedData.frameNumber]]);
        largeCost = 1e6;
        
        for frame = uniqueFrames
            gtObjects = groundTruthData([groundTruthData.frameNumber] == frame);
            detectedObjects = detectedData([detectedData.frameNumber] == frame);
            numGt = length(gtObjects);
            numDet = length(detectedObjects);
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
        end
        
        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        
        negPrecision = -precision;  % We minimize negative precision to maximize precision
        negRecall = -recall;        % We minimize negative recall to maximize recall
    end

% Load previous state if available
stateFile = 'output/gamultiobj_state.mat';
if isfile(stateFile)
    load(stateFile, 'state');
    options = state.options;
else
    options = optimoptions('gamultiobj', ...
        'PopulationSize', 1000, ...  % Increase population size
        'MaxGenerations', 5000, ...  % Allow more generations
        'FunctionTolerance', 1e-6, ...  % Set a low function tolerance
        'MaxStallGenerations', 500, ...  % Increase max stall generations
        'UseParallel', true, ...  % Enable parallel computation
        'ParetoFraction', 0.7, ...  % Keep a larger fraction on the Pareto front
        'Display', 'iter', ...  % Display output at each iteration
        'OutputFcn', @saveCheckpoint ...  % Custom function to save output periodically
        );
end

% Define the objective function and variable bounds
FitnessFunction = @evaluateParams;
numberOfVariables = 12;
lb = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0];
ub = [5, 100, 100, 10, 10, 10, 10, 11, 10, 20, 10, 10];

% Run the optimization
[x, Fval, exitFlag, Output] = gamultiobj(FitnessFunction, numberOfVariables, [], [], [], [], lb, ub, options);

% Save the final results
save('output/final_pareto_solutions.mat', 'x', 'Fval', 'exitFlag', 'Output');

% Plot the Pareto front
figure;
plot(Fval(:,1), Fval(:,2), 'bo');
xlabel('Precision');
ylabel('Recall');
title('Pareto Front Video 1');

% Function to save output periodically and for checkpointing
    function [state, options, optchanged] = saveCheckpoint(options, state, flag)
        optchanged = false;
        if strcmp(flag, 'iter')
            save('output/gamultiobj_state.mat', 'state');
            fprintf('Checkpoint saved at generation %d.\n', state.Generation);
        end
    end
end