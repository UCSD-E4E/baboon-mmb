function optimize()
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
    
    % Nested function: evaluation
    function f1Score = evaluateParams(params)
        % Call the baboon_mmb function with given parameters
        detectedData = baboon_mmb('K', params.K, 'CONNECTIVITY', 8, ...
                                  'AREA_MIN', params.AREA_MIN, 'AREA_MAX', params.AREA_MAX, ...
                                  'ASPECT_RATIO_MIN', params.ASPECT_RATIO_MIN, 'ASPECT_RATIO_MAX', params.ASPECT_RATIO_MAX, ...
                                  'L', params.L, 'KERNEL', 3, 'BITWISE_OR', 0, ...
                                  'PIPELINE_LENGTH', params.PIPELINE_LENGTH, 'PIPELINE_SIZE', params.PIPELINE_SIZE, ...
                                  'H', params.H, 'MAX_NITER_PARAM', params.MAX_NITER_PARAM, ...
                                  'GAMMA1_PARAM', params.GAMMA1_PARAM, 'GAMMA2_PARAM', params.GAMMA2_PARAM, ...
                                  'FRAME_RATE', 10, 'IMAGE_SEQUENCE', 'input/viso_video_1');
        
        % Load ground truth data
        groundTruthData = loadGroundTruth('input/viso_video_1_gt.txt');
    
        % Initialize counters
        TP = 0; FP = 0; FN = 0;
        
        % Handle case where detectedData is empty
        if isempty(detectedData)
            detectedData = struct('frameNumber', [], 'id', [], 'x', [], 'y', [], 'width', [], 'height', [], 'cx', [], 'cy', []);
        end
        
        % Unique frames
        uniqueFrames = unique([groundTruthData.frameNumber, [detectedData.frameNumber]]);
        largeCost = 1e6;
        
        % Compare ground truth and detected data
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
        
        % Compute precision, recall, and F1 score with zero division checks
        if TP + FP == 0
            precision = 0;
        else
            precision = TP / (TP + FP);
        end
        
        if TP + FN == 0
            recall = 0;
        else
            recall = TP / (TP + FN);
        end
        
        if precision + recall == 0
            f1Score = 0;
        else
            f1Score = 2 * (precision * recall) / (precision + recall);
        end
            
        % Save precision and recall to a text file
        fileID = fopen('output/precision_recall_data.txt', 'a');
        fprintf(fileID, '%f %f\n', precision, recall);
        fclose(fileID);
        
        % Return negative F1 score for minimization
        f1Score = -f1Score;
    end
    
    % Define the feasibility check function
    function feasible = isFeasible(params)
        feasible = params.AREA_MIN <= params.AREA_MAX && ...
                   params.ASPECT_RATIO_MIN <= params.ASPECT_RATIO_MAX && ...
                   params.H <= params.PIPELINE_LENGTH && ...
                   params.GAMMA1_PARAM <= params.GAMMA2_PARAM;
    end
    
    % Define the optimization variables
    vars = [
        optimizableVariable('K', [0, 5], 'Type', 'real');
        optimizableVariable('AREA_MIN', [0, 100], 'Type', 'integer');
        optimizableVariable('AREA_MAX', [0, 100], 'Type', 'integer');
        optimizableVariable('ASPECT_RATIO_MIN', [0, 10], 'Type', 'real');
        optimizableVariable('ASPECT_RATIO_MAX', [0, 10], 'Type', 'real');
        optimizableVariable('L', [1, 10], 'Type', 'integer');
        optimizableVariable('PIPELINE_LENGTH', [1, 10], 'Type', 'integer');
        optimizableVariable('PIPELINE_SIZE', [3, 11], 'Type', 'integer');
        optimizableVariable('H', [1, 10], 'Type', 'integer');
        optimizableVariable('MAX_NITER_PARAM', [1, 20], 'Type', 'integer');
        optimizableVariable('GAMMA1_PARAM', [0, 1], 'Type', 'real');
        optimizableVariable('GAMMA2_PARAM', [0, 1], 'Type', 'real');
    ];
    
    % Define the objective function for Bayesian optimization
    objFunc = @(params)feasibleObjFunc(params);
    
    % Wrapper for objective function with feasibility check
    function f = feasibleObjFunc(params)
        if isFeasible(params)
            f = evaluateParams(params);
        else
            f = inf; % Penalize infeasible points
        end
    end
    

    results = bayesopt(objFunc, vars, ...
                   'AcquisitionFunctionName', 'expected-improvement-plus', ...
                   'Verbose', 2, ...  % Set verbosity to display minimal information
                   'MaxObjectiveEvaluations', 1e6, ...
                   'PlotFcn', [], ...  % Disable all plotting
                   'UseParallel', false); 

    % Save the results
    save('output/optimization_results.mat', 'results');
end