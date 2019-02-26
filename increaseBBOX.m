function  nBBOX = increaseBBOX(BBOX, pixels)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

nBBOX = zeros(size(BBOX));
rows = size(BBOX, 1);

for i = 1:rows

    nBBOX(i, 1) = BBOX(i, 1) - pixels;
    nBBOX(i, 2) = BBOX(i, 2) - pixels;
    nBBOX(i, 3) = BBOX(i, 3) + (2 * pixels);
    nBBOX(i, 4) = BBOX(i, 4) + (2 * pixels);

end




end

