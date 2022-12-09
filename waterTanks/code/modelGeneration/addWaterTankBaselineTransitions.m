function [lattice] = addWaterTankBaselineTransitions(lattice,N,deltawl,inflow,outflow,rounding)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
format long
latticeSize = size(lattice);

for i=1:latticeSize(2)
    tempCell = lattice(i);
    wl = tempCell.wl;
    newMapKey = N;
    
    clear newMapValue
    for K=0:1:N
        [nextwls] = computeWaterTankBaselineTransition(wl,K,deltawl,inflow,outflow,10);

        if(rounding==1)
            [nextwls] = scrubTransitionsWorstCase(nextwls,deltawl);
        end
        
        latticeTransitions = [];
        for iter = 1:length(nextwls)
            nextwl = max(nextwls(iter),0);
            nextwlid = ceil(nextwl/deltawl);

            if(length(latticeTransitions)==0)
                latticeTransitions = [latticeTransitions; nextwlid 0];
            else
                I = sum(latticeTransitions(:, 1) == nextwlid);
                if(I==0)
                    latticeTransitions = [latticeTransitions; nextwlid 0];
                end
            end
        end
        newMapValue{K+1}=latticeTransitions;
    end
    tempCell.nextPoints(newMapKey)=newMapValue;
    lattice(i)=tempCell;

end
end

