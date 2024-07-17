function optimize(varargin)
% Parse command line arguments
p = inputParser;
validPath = @(x) ischar(x) && isfolder(x);
validFile = @(x) ischar(x) && isfile(x);
validStringNum = @(x) ischar(x) && ~isnan(str2double(x));

% All parameters are added as strings
addParameter(p, 'InputPath', 'input/viso_video_1', validPath);
addParameter(p, 'GroundTruthPath', 'input/viso_video_1_gt.txt', validFile);
addParameter(p, 'FrameRate', '10', validStringNum);
addParameter(p, 'PopulationSize', '1000', validStringNum);
addParameter(p, 'MaxGenerations', '5000', validStringNum);
addParameter(p, 'FunctionTolerance', '1e-6', validStringNum);
addParameter(p, 'MaxStallGenerations', '500', validStringNum);
addParameter(p, 'UseParallel', 'true', @(x) ischar(x) && any(strcmpi(x, {'true', 'false', '0', '1'})));
addParameter(p, 'ParetoFraction', '0.7', validStringNum);
addParameter(p, 'Display', 'iter', @ischar);
addParameter(p, 'Continue', 'false', @(x) ischar(x) && any(strcmpi(x, {'true', 'false', '0', '1'})));
addParameter(p, 'OptimizationType', '2', @(x) ischar(x) && any(str2double(x) == [1, 2, 3, 4]));

parse(p, varargin{:});

% Convert parameters to their appropriate types
results.InputPath = p.Results.InputPath;
results.GroundTruthPath = p.Results.GroundTruthPath;
results.FrameRate = str2double(p.Results.FrameRate);
results.PopulationSize = str2double(p.Results.PopulationSize);
results.MaxGenerations = str2double(p.Results.MaxGenerations);
results.FunctionTolerance = str2double(p.Results.FunctionTolerance);
results.MaxStallGenerations = str2double(p.Results.MaxStallGenerations);
results.UseParallel = strcmpi(p.Results.UseParallel, 'true') || str2double(p.Results.UseParallel) == 1;
results.ParetoFraction = str2double(p.Results.ParetoFraction);
results.Display = p.Results.Display;
results.Continue = strcmpi(p.Results.Continue, 'true') || str2double(p.Results.Continue) == 1;
results.OptimizationType = str2double(p.Results.OptimizationType);

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
    function [precision, recall] = evaluateParams(params)
        detectedData = baboon_mmb('K', params(1), 'CONNECTIVITY', 8, ...
            'AREA_MIN', params(2), 'AREA_MAX', params(3), ...
            'ASPECT_RATIO_MIN', params(4), 'ASPECT_RATIO_MAX', params(5), ...
            'L', params(6), 'KERNEL', 3, 'BITWISE_OR', false, ...
            'PIPELINE_LENGTH', params(7), 'PIPELINE_SIZE', params(8), ...
            'H', params(9), 'MAX_NITER_PARAM', params(10), ...
            'GAMMA1_PARAM', params(11), 'GAMMA2_PARAM', params(12), ...
            'FRAME_RATE', results.FrameRate, 'IMAGE_SEQUENCE', results.InputPath, 'DEBUG', false);
        
        groundTruthData = loadGroundTruth(results.GroundTruthPath);
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
            
            switch results.OptimizationType
                case 1
                    for i = 1:numGt
                        gtOverlap = false;
                        for j = 1:numDet
                            bbGt = [gtObjects(i).x, gtObjects(i).y, gtObjects(i).width, gtObjects(i).height];
                            bbDet = [detectedObjects(j).x, detectedObjects(j).y, detectedObjects(j).width, detectedObjects(j).height];
                            overlapRatio = bboxOverlapRatio(bbGt, bbDet);
                            if overlapRatio > 0
                                gtOverlap = true;
                                break;
                            end
                        end
                        if gtOverlap
                            TP = TP + 1;
                        else
                            FN = FN + 1;
                        end
                    end
                    FP = numDet - TP;
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
                            TP = TP + 1
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
                                    TP = TP + 1
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
        
        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        f1Score = (2 * precision * recall) / (precision + recall);
        
        % Save parameters, precision, and recall to a text file
        if f1Score > 0
            if results.UseParallel
                workerID = getCurrnentTask().ID;
                resultsFile = sprintf('output/evaluation_results_worker_%d.txt', workerID);
                paramStr = sprintf('%.4f ', params);
                fileID = fopen(resultsFile, 'a');
                fprintf(fileID, '%s Precision: %.4f Recall: %.4f F1: %.4f\n', paramStr, precision, recall, f1Score);
                fclose(fileID);
            else
                resultsFile = 'output/evaluation_results.txt';
                paramStr = sprintf('%.4f ', params);
                fileID = fopen(resultsFile, 'a');
                fprintf(fileID, '%s Precision: %.4f Recall: %.4f F1: %.4f\n', paramStr, precision, recall, f1Score);
                fclose(fileID);
            end
        end
    end

% Load checkpoint if available
stateFile = 'output/gamultiobj_state.mat';
if results.Continue && isfile(stateFile)
    load(stateFile, 'state', 'options');
    fprintf('Continuing from saved state...\n');
    % Adjust MaxGenerations based on remaining generations
    options.MaxGenerations = results.MaxGenerations - state.Generation;
else
    % Use command line arguments
    options = optimoptions('gamultiobj', ...
        'PopulationSize', results.PopulationSize, ...
        'MaxGenerations', results.MaxGenerations, ...
        'FunctionTolerance', results.FunctionTolerance, ...
        'MaxStallGenerations', results.MaxStallGenerations, ...
        'UseParallel', results.UseParallel, ...
        'ParetoFraction', results.ParetoFraction, ...
        'Display', results.Display, ...
        'OutputFcn', @saveCheckpoint);
end

% Define the objective function and variable bounds
FitnessFunction = @evaluateParams;
numberOfVariables = 12;
lb = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0];
ub = [8, 100, 100, 10, 10, 10, 10, 11, 10, 20, 1, 1];
intIndices = [6, 7, 8, 9, 10];

% Run the optimization
[x, Fval, exitFlag, Output, state] = gamultiobj(FitnessFunction, numberOfVariables, [], [], [], [], lb, ub, [], intIndices, options);

% Save the final results
save('output/final_pareto_solutions.mat', 'x', 'Fval', 'exitFlag', 'Output');

if results.UseParallel
    resultFiles = dir('output/evaluation_results_worker_*.txt');
    mergedResults = 'output/evaluation_results.txt';
    mergedFileId = fopen(mergedResults, 'a');
    for k = 1:length(resultFiles)
        workerFile = fullfile(resultFile(k).folder, resultFiles(k).name);
        workerData = fileread(workerFile);
        fprintf(mergedFileId, '%s\n', workerData);
    end
    fclose(mergedFileId);
end

% Save the Pareto front plot to a file
figure;
plot(Fval(:,1), Fval(:,2), 'bo');
xlabel('Precision');
ylabel('Recall');
title('Pareto Front Video 1');
saveas(gcf, 'output/pareto_front.png');

% Function to save output periodically and for checkpointing
    function [state, options, optchanged] = saveCheckpoint(options, state, flag)
        optchanged = false;
        if strcmp(flag, 'iter') || strcmp(flag, 'diagnose')
            save('output/gamultiobj_state.mat', 'state', 'options');
        end
    end
end