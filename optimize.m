% Define the parameters for the function call
params = struct();
params.K = 4;
params.CONNECTIVITY = 8;
params.AREA_MIN = 5;
params.AREA_MAX = 80;
params.ASPECT_RATIO_MIN = 1.0;
params.ASPECT_RATIO_MAX = 6.0;
params.L = 4;
params.KERNEL = 3;
params.BITWISE_OR = false;
params.PIPELINE_LENGTH = 5;
params.PIPELINE_SIZE = 7;
params.H = 3;
params.MAX_NITER_PARAM = 10;
params.GAMMA1_PARAM = 8;
params.GAMMA2_PARAM = 8;
params.FRAME_RATE = 10;
params.IMAGE_SEQUENCE = 'path/to/your/image_sequence';

% Call the function and retrieve the object data
detectedData = baboon_mmb('K', params.K, 'CONNECTIVITY', params.CONNECTIVITY, ...
                     'AREA_MIN', params.AREA_MIN, 'AREA_MAX', params.AREA_MAX, ...
                     'ASPECT_RATIO_MIN', params.ASPECT_RATIO_MIN, 'ASPECT_RATIO_MAX', params.ASPECT_RATIO_MAX, ...
                     'L', params.L, 'KERNEL', params.KERNEL, 'BITWISE_OR', params.BITWISE_OR, ...
                     'PIPELINE_LENGTH', params.PIPELINE_LENGTH, 'PIPELINE_SIZE', params.PIPELINE_SIZE, ...
                     'H', params.H, 'MAX_NITER_PARAM', params.MAX_NITER_PARAM, ...
                     'GAMMA1_PARAM', params.GAMMA1_PARAM, 'GAMMA2_PARAM', params.GAMMA2_PARAM, ...
                     'FRAME_RATE', params.FRAME_RATE, 'IMAGE_SEQUENCE', params.IMAGE_SEQUENCE);

% Load ground truth data from file
groundTruthFile = load('input/viso_video_1_gt.txt');

% Define the structure template
template = struct('frameNumber', [], 'id', [], 'x', [], 'y', [], 'width', [], 'height', [], 'cx', [], 'cy', []);

% Preallocate the structure array
numEntries = size(groundTruthFile, 1);
groundTruthData = repmat(template, numEntries, 1);

% Populate the structure array
for i = 1:numEntries
    groundTruthData(i).frameNumber = groundTruthFile(i, 1);
    groundTruthData(i).id = groundTruthFile(i, 2);
    groundTruthData(i).x = groundTruthFile(i, 3);
    groundTruthData(i).y = groundTruthFile(i, 4);
    groundTruthData(i).width = groundTruthFile(i, 5);
    groundTruthData(i).height = groundTruthFile(i, 6);
    groundTruthData(i).cx = groundTruthFile(i, 3) + groundTruthFile(i, 5) / 2; % Calculate center x
    groundTruthData(i).cy = groundTruthFile(i, 4) + groundTruthFile(i, 6) / 2; % Calculate center y
end

TP = 0;
FP = 0;
FN = 0;

uniqueFrames = unique([groundTruthData.frameNumber, [detectedData.frameNumber]]);

% Set a large but finite cost for non-overlapping assignments
largeCost = 1e6;

for frame = uniqueFrames
    % Get bojects from the current frame
    gtObjects = groundTruthData([groundTruthData.frameNumber] == frame);
    detectedObjects = detectedData([detectedData.frameNumber] == frame);

    % Create cost matrix for object matching
    numGt = length(gtObjects);
    numDet = length(detectedObjects);
    costMatrix = largeCost * ones(numGt, numDet);

    for i = 1:numGt
        for j = 1:numDet
            % Calculate IoU (Intersection over Union)
            bbGt = [gtObjects(i).x, gtObjects(i).y, gtObjects(i).width, gtObjects(i).height];
            bbDet = [detectedObjects(j).x, detectedObjects(j).y, detectedObjects(j).width, detectedObjects(j).height];
            overlapRatio = bboxOverlapRatio(bbGt, bbDet);
            if overlapRatio > 0
                costMatrix(i, j) = 1 - overlapRatio;
            end
        end
    end

    % Perform optimal assignment using the Hungarian algorithm
    [assignments, unassignedRows, unassignedCols] = assignDetectionsToTracks(costMatrix, largeCost - 1);

    % Update TP, FP, FN counts
    TP = TP + size(assignments, 1);
    FP = FP + length(unassignedCols);
    FN = FN + length(unassignedRows);
end

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1Score = 2 * (precision * recall) / (precision + recall);

% Display the evaluation metrics
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1Score);