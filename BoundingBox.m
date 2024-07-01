classdef BoundingBox
    properties
        x
        y
        w
        h
    end
    
    methods
        function obj = BoundingBox(x, y, w, h)
            obj.x = int32(x);
            obj.y = int32(y);
            obj.w = int32(w);
            obj.h = int32(h);
        end
    end
end
