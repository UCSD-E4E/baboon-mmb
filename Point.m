classdef Point
    properties
        x
        y
    end
    
    methods
        function obj = Point(x, y)
            obj.x = int32(x);
            obj.y = int32(y);
        end
    end
end
