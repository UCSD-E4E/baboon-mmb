function output = amfd(K, CONNECTIVITY, AREA_MIN, AREA_MAX, ASPECT_RATIO_MIN, ASPECT_RATIO_MAX, KERNEL, grayFrames)
% fprintf('Processing frames using AMFD...\n');

% Validate input
if isempty(grayFrames) || ~iscell(grayFrames)
    error('Invalid input: grayFrames must be a non-empty cell array of grayscale images');
end

numFrames = numel(grayFrames);
[height, width] = size(grayFrames{1});

% Ensure all frames are the same size
if any(cellfun(@(x) ~isequal(size(x), [height, width]), grayFrames))
    error('Invalid input: All frames must have the same dimensions.');
end

output = cell(1, numFrames);
output{1} = zeros(height, width, 'uint8');  % Handle edge frames
output{numFrames} = zeros(height, width, 'uint8');

% Preallocate and reuse the structuring element
se = strel('disk', double(max(1, floor(KERNEL/2))));

% Precompute grayscale frames to double format for all frames
grayFramesDbl = cellfun(@double, grayFrames, 'UniformOutput', false);

for t = 2:(numFrames - 1)
    I_prev = grayFramesDbl{t-1};
    I_curr = grayFramesDbl{t};
    I_next = grayFramesDbl{t+1};
    
    % Calculate difference images and accumulate
    D_t1 = abs(I_curr - I_prev);
    D_t2 = abs(I_next - I_prev);
    D_t3 = abs(I_next - I_curr);
    Id = (D_t1 + D_t2 + D_t3) / 3;
    
    % Threshold calculation
    mu = mean(Id(:));
    sigma = std(Id(:));
    T = mu + K * sigma;
    
    % Binarize and morphological operations
    binaryImage = imclose(imopen(Id >= T, se), se);
    
    % Connected components and properties
    labeledImage = bwlabel(binaryImage, double(CONNECTIVITY));
    props = regionprops(labeledImage, 'Area', 'BoundingBox');
    
    for propIdx = 1:length(props)
        bb = props(propIdx).BoundingBox;
        aspectRatio = max(bb(3), bb(4)) / min(bb(3), bb(4));
        if props(propIdx).Area < AREA_MIN || props(propIdx).Area > AREA_MAX || aspectRatio < ASPECT_RATIO_MIN || aspectRatio > ASPECT_RATIO_MAX
            binaryImage(labeledImage == propIdx) = 0;
        end
    end
    
    % Store the refined binary mask
    output{t} = binaryImage;
end
end