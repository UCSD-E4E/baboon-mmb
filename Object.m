classdef Object
    properties
        center
        bounding_box
    end
    
    methods
        function obj = Object(center, bounding_box)
            obj.center = center;
            obj.bounding_box = bounding_box;
        end
    end
end
