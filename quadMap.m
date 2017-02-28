function xOut = quadMap(xIn)

xOut = [xIn, xIn.^2, xIn(:,1).*xIn(:,2)];
end
