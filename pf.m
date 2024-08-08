function output = pf(PIPELINE_LENGTH, PIPELINE_SIZE, H, combinedMasks)
    % fprintf('Running the Pipeline Filter...\n');

    % Buffer to store recent frame data
    buffer = cell(PIPELINE_LENGTH + 1, 1);
    totalFrames = numel(combinedMasks);
    output = [];

    % Precompute frame data for all frames
    frameData = cell(totalFrames, 1);
    for i = 1:totalFrames
        frameData{i} = computeFrameData(combinedMasks{i});
    end

    % Initialize buffer with the first set of frames
    buffer(1:min(PIPELINE_LENGTH + 1, totalFrames)) = frameData(1:min(PIPELINE_LENGTH + 1, totalFrames));

    % Process each frame
    for currentFrame = 1:totalFrames
        % Assign unique IDs to objects in the first buffer slot
        if ~isempty(buffer{1})
            for objIdx = 1:numel(buffer{1})
                buffer{1}(objIdx).id = objIdx;
            end
        end

        % Reset all other IDs to -1
        for bufIdx = 2:numel(buffer)
            if ~isempty(buffer{bufIdx})
                [buffer{bufIdx}.id] = deal(-1);
            end
        end

        % Optimal assignment for subsequent buffer slots
        for bufIdx = 2:numel(buffer)
            if ~isempty(buffer{bufIdx}) && ~isempty(buffer{1})
                costMatrix = computeCostMatrix(buffer{1}, buffer{bufIdx});
                largeCost = 1e6;  % A large cost for unmatched assignments
                assignments = matchpairs(costMatrix, largeCost);

                for objIdx = 1:size(assignments, 1)
                    if assignments(objIdx, 2) > 0
                        % Get the matched objects
                        obj1 = buffer{1}(assignments(objIdx, 1));
                        obj2 = buffer{bufIdx}(assignments(objIdx, 2));

                        % Check the conditions before assigning the ID
                        if 0 < abs(obj1.cx - obj2.cx) && abs(obj1.cx - obj2.cx) < PIPELINE_SIZE && 0 < abs(obj1.cy - obj2.cy) && abs(obj1.cy - obj2.cy) < PIPELINE_SIZE
                            buffer{bufIdx}(assignments(objIdx, 2)).id = obj1.id;
                        end
                    end
                end
            end
        end
        
        % Check for objects in buffer{1} that have at least H matching objects in buffers {2} to {numel(buffer)}
        for objIdx = 1:numel(buffer{1})
            obj1 = buffer{1}(objIdx);
            matchCount = 0;
            matchBuffers = false(1, numel(buffer) - 1);

            for bufIdx = 2:numel(buffer)
                if ~isempty(buffer{bufIdx}) && any([buffer{bufIdx}.id] == obj1.id)
                    matchCount = matchCount + 1;
                    matchBuffers(bufIdx - 1) = true;
                end
            end

            if matchCount >= H
                obj1.frameNumber = currentFrame;
                output = [output, obj1]; %#ok<AGROW>

                % Interpolate positions for missing assignments
                for bufIdx = 2:numel(buffer)
                    if ~matchBuffers(bufIdx - 1)
                        % Linear interpolation based on surrounding frames
                        prevBuffer = find(matchBuffers(1:bufIdx-1), 1, 'last') + 1;
                        nextBuffer = find(matchBuffers(bufIdx:end), 1, 'first') + bufIdx;
                        
                        if isempty(prevBuffer)
                            % Use obj1 itself for previous data if no previous buffer
                            prevBuffer = 1;
                            prevObj = obj1;
                        else 
                            prevObj = buffer{prevBuffer}(arrayfun(@(x) x.id == obj1.id, buffer{prevBuffer}));
                        end

                        if ~isempty(nextBuffer)
                            nextObj = buffer{nextBuffer}(arrayfun(@(x) x.id == obj1.id, buffer{nextBuffer}));

                            % Interpolate position 
                            alpha = (bufIdx - prevBuffer) / (nextBuffer - prevBuffer);
                            interpolatedObj = obj1;
                            interpolatedObj.x = round((1 - alpha) * prevObj.x + alpha * nextObj.x);
                            interpolatedObj.y = round((1 - alpha) * prevObj.y + alpha * nextObj.y);
                            interpolatedObj.width = round((1 - alpha) * prevObj.width + alpha * nextObj.width);
                            interpolatedObj.height = round((1 - alpha) * prevObj.height + alpha * nextObj.height);
                            interpolatedObj.cx = (1 - alpha) * prevObj.cx + alpha * nextObj.cx;
                            interpolatedObj.cy = (1 - alpha) * prevObj.cy + alpha * nextObj.cy;

                            buffer{bufIdx} = [buffer{bufIdx}, interpolatedObj]; 
                        end
                    end
                end
            end
        end

        % Shift buffer and add new frame data
        buffer(1:end-1) = buffer(2:end);
        nextFrameIdx = currentFrame + PIPELINE_LENGTH + 1;
        if nextFrameIdx <= totalFrames
            buffer{end} = computeFrameData(combinedMasks{nextFrameIdx});
        else
            buffer{end} = computeFrameData(zeros(size(combinedMasks{1})));
        end
    end
end

function frameData = computeFrameData(frame)
    labeledImage = bwlabel(frame);
    props = regionprops(labeledImage, 'BoundingBox');
    numProps = numel(props);

    % Initialize frameData structure
    if numProps == 0
        frameData = struct('id', {}, 'frameNumber', {}, 'x', {}, 'y', {}, 'width', {}, 'height', {}, 'cx', {}, 'cy', {});
    else
        frameData = struct('id', -1, 'frameNumber', -1, 'x', [], 'y', [], 'width', [], 'height', [], 'cx', [], 'cy', []);
        for i = 1:numProps
            bb = floor(props(i).BoundingBox);
            frameData(i).x = bb(1);
            frameData(i).y = bb(2);
            frameData(i).width = bb(3);
            frameData(i).height = bb(4);
            frameData(i).cx = bb(1) + bb(3) / 2;
            frameData(i).cy = bb(2) + bb(4) / 2;
        end
    end
end

function costMatrix = computeCostMatrix(objects1, objects2)
    numObjects1 = numel(objects1);
    numObjects2 = numel(objects2);
    costMatrix = zeros(numObjects1, numObjects2);

    for i = 1:numObjects1
        for j = 1:numObjects2
            % Using Euclidean distance as the cost metric
            costMatrix(i, j) = sqrt((objects1(i).x - objects2(j).x)^2 + (objects1(i).y - objects2(j).y)^2);
        end
    end
end