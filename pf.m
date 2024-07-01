function output = pf(PIPELINE_LENGTH, PIPELINE_SIZE, H, combinedMasks)
    fprintf('Running the Pipeline Filter...\n');
    startTime = tic;

    pipeline_observations = cell(1, PIPELINE_LENGTH + 1);
    output = {};

    frame_count = numel(combinedMasks);

    for frame_idx = 1:frame_count
        for objects_idx = 1:min(PIPELINE_LENGTH, frame_count - frame_idx + 1)
            if isempty(pipeline_observations{objects_idx})
                if frame_idx + objects_idx - 1 <= frame_count
                    pipeline_observations{objects_idx} = get_objects(combinedMasks{frame_idx + objects_idx - 1});
                else
                    pipeline_observations{objects_idx} = {};
                end
            end
        end

        if ~isempty(pipeline_observations{1})
            object_in_next_frames = find_object_pairs(pipeline_observations, PIPELINE_LENGTH, PIPELINE_SIZE);

            for object_idx = 1:numel(pipeline_observations{1})
                object = pipeline_observations{1}{object_idx};
                h = sum(object_in_next_frames(object_idx, :) ~= -1);

                if h >= H && h ~= PIPELINE_LENGTH
                    pointline = cell(1, PIPELINE_LENGTH + 1);
                    pointline{1} = pipeline_observations{1}{object_idx};

                    for pipeline_frame_idx = 2:PIPELINE_LENGTH + 1
                        if object_in_next_frames(object_idx, pipeline_frame_idx - 1) ~= -1
                            next_frame_idx = object_in_next_frames(object_idx, pipeline_frame_idx - 1) + 1;
                            pointline{pipeline_frame_idx} = pipeline_observations{pipeline_frame_idx}{next_frame_idx};
                        end
                    end

                    for pipeline_frame_idx = 2:PIPELINE_LENGTH
                        if isempty(pointline{pipeline_frame_idx})
                            prev_point_idx = find(~cellfun(@isempty, pointline(1:pipeline_frame_idx - 1)), 1, 'last');
                            next_point_idx = find(~cellfun(@isempty, pointline(pipeline_frame_idx + 1:end)), 1, 'first') + pipeline_frame_idx - 1;

                            if ~isempty(next_point_idx) && ~isempty(prev_point_idx)
                                prev_point = pointline{prev_point_idx};
                                next_point = pointline{next_point_idx};
                                interp_center_x = floor((prev_point.center.x + next_point.center.x) / 2);
                                interp_center_y = floor((prev_point.center.y + next_point.center.y) / 2);
                                interp_bbox_x = floor((prev_point.bounding_box.x + next_point.bounding_box.x) / 2);
                                interp_bbox_y = floor((prev_point.bounding_box.y + next_point.bounding_box.y) / 2);
                                interp_bbox_w = floor((prev_point.bounding_box.w + next_point.bounding_box.w) / 2);
                                interp_bbox_h = floor((prev_point.bounding_box.h + next_point.bounding_box.h) / 2);
                                interp_center = Point(interp_center_x, interp_center_y);
                                interp_bbox = BoundingBox(interp_bbox_x, interp_bbox_y, interp_bbox_w, interp_bbox_h);
                                pointline{pipeline_frame_idx} = Object(interp_center, interp_bbox);
                            end
                        end
                    end
                elseif h == PIPELINE_LENGTH
                else
                    pipeline_observations{1}{object_idx} = [];
                end
            end

            for object_idx = 1:numel(pipeline_observations{1})
                object = pipeline_observations{1}{object_idx};
                if ~isempty(object)
                    output{end + 1} = {frame_idx, object.bounding_box.x, object.bounding_box.y, object.bounding_box.w, object.bounding_box.h};
                end
            end
        end

        shiftObservations(pipeline_observations, PIPELINE_LENGTH);

        printProgressBar(frame_idx, frame_count, startTime);
    end
end

function objects = get_objects(mask)
    CC = bwconncomp(mask);
    stats = regionprops(CC, 'Centroid', 'BoundingBox');
    objects = cell(1, length(stats));
    for i = 1:length(stats)
        center = Point(stats(i).Centroid(1), stats(i).Centroid(2));
        bounding_box = BoundingBox(stats(i).BoundingBox(1), stats(i).BoundingBox(2), ...
                                   stats(i).BoundingBox(3), stats(i).BoundingBox(4));
        objects{i} = Object(center, bounding_box);
    end
end

function object_in_next_frames = find_object_pairs(pipeline_observations, PIPELINE_LENGTH, PIPELINE_SIZE)
    num_objects = numel(pipeline_observations{1});
    object_in_next_frames = -ones(num_objects, PIPELINE_LENGTH);

    for pipeline_frame_idx = 2:PIPELINE_LENGTH + 1
        cost_matrix = zeros(num_objects, numel(pipeline_observations{pipeline_frame_idx}));
        for a_idx = 1:num_objects
            for b_idx = 1:numel(pipeline_observations{pipeline_frame_idx})
                a = pipeline_observations{1}{a_idx};
                b = pipeline_observations{pipeline_frame_idx}{b_idx};
                SabX = abs(a.center.x - b.center.x);
                SabY = abs(a.center.y - b.center.y);
                if (SabX > 0 && SabX < PIPELINE_SIZE && SabY > 0 && SabY < PIPELINE_SIZE)
                    cost_matrix(a_idx, b_idx) = 1;
                end
            end
        end
        [assignment, cost] = assignmentoptimal(cost_matrix);
        for a_idx = 1:num_objects
            b_idx = assignment(a_idx);
            if b_idx ~= 0 && cost_matrix(a_idx, b_idx) == 1
                object_in_next_frames(a_idx, pipeline_frame_idx - 1) = b_idx - 1; % Adjust index for MATLAB 1-based indexing
            end
        end
    end
end

function shiftObservations(pipeline_observations, PIPELINE_LENGTH)
    % Shift the pipeline observations
    for objects_idx = 1:PIPELINE_LENGTH
        pipeline_observations{objects_idx} = pipeline_observations{objects_idx + 1};
    end

    % Clear the last entry in the pipeline observations
    pipeline_observations{PIPELINE_LENGTH + 1} = {};
end

function printProgressBar(currentStep, totalSteps, startTime)
    % Calculate percentage completion
    percentage = 100 * (currentStep / totalSteps);
    barLength = floor(50 * (currentStep / totalSteps));  % Length of the progress bar in characters
    bar = repmat('#', 1, barLength);  % Create the progress bar
    spaces = repmat(' ', 1, 50 - barLength);  % Spaces to fill the rest of the bar
    
    % Calculate elapsed time and estimate remaining time
    elapsedTime = toc(startTime);
    remainingTime = elapsedTime / currentStep * (totalSteps - currentStep);
    
    % Format remaining time as HH:MM:SS
    hours = floor(remainingTime / 3600);
    mins = floor(mod(remainingTime, 3600) / 60);
    secs = floor(mod(remainingTime, 60));
    
    % Clear the previous line before printing new progress information
    if currentStep > 1  % Avoid clearing if it's the first step
        fprintf('\033[A\033[K');  % Move cursor up one line and clear line
    end
    
    % Print progress bar with time estimate
    fprintf('[%s%s] %3.0f%% - Elapsed: %02d:%02d:%02d, Remaining: %02d:%02d:%02d\n', ...
            bar, spaces, percentage, ...
            floor(elapsedTime / 3600), mod(floor(elapsedTime / 60), 60), mod(elapsedTime, 60), ...
            hours, mins, secs);

    if currentStep == totalSteps
        fprintf('\n');  % Move to the next line after completion
    end
end
