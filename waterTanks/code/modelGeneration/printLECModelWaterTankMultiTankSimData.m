function[] = printLECModelWaterTankMultiTank(lattice, maxN, fid,deltawl,numTanks)


latticeSize=size(lattice);
wlidMax=latticeSize(2);

fprintf(fid, '\n \n');


% LECProbs = [0.020500000000000   0.024900000000000   0.036800000000000   0.053500000000000   0.076400000000000   0.095200000000000   0.102800000000000   0.093200000000000   0.073300000000000   0.052700000000000   0.040700000000000   0.023000000000000   0.020200000000000];
% LECLowerProb = 0.143200000000000;
% LECUpperProb = 0.143600000000000;

%            -6      -5     -4     -3     -2    -1      0       1     2       3      4     5
% LECProbs = [0.0017 0.0267 0.0450 0.1083 0.1800 0.2017 0.1983 0.1217 0.0683 0.0267 0.0183 0.0033];

% deltawl=5
if deltawl==5
    % deltawl=5
    LECProbs = [0.1817 0.77 0.0483];
elseif deltawl==2
    % deltawl=2
    LECProbs = [(0.0017+0.0267) (0.0450+0.1083) (0.1800+0.2017) (0.1983+0.1217) (0.0683+0.0267) (0.0183+0.0033)];
elseif deltawl==1
    LECProbs = [0.0017 0.0267 0.0450 0.1083 0.1800 0.2017 0.1983 0.1217 0.0683 0.0267 0.0183 0.0033];
else
    error('LEC model not hardcoded for deltawl=%i',deltawl)
end

% sum(LECProbs)
assert(abs(sum(LECProbs)-1)<= 0.0001,'LEC error probabilities do not sum to 1');

numBins = length(LECProbs);

for j=1:numTanks-1
    for k=1:wlidMax
        currcell=lattice(k);
        wlid=currcell.wlid;
        wl = currcell.wl;

        fprintf(fid, '    [] currN=0&sink=0&tankFlag=%i&wlid%i=%i -> ',j,j,wlid);
%         fprintf(fid, "%i:(wlidPer%i'=0)&(tankFlag'=%i) + %i:(wlidPer%i'=wlidMax)&(tankFlag'=%i) + ",LECLowerProb,j,j+1,LECUpperProb,j,j+1);
        for i=1:numBins
            nextwl = wl+i*deltawl-deltawl*(numBins+1)/2;
            nextwlid = ceil(nextwl/deltawl);
            nextwlid = max(0,nextwlid);
            nextwlid = min(wlidMax-1,nextwlid);
            if i==numBins
                fprintf(fid, ' %i:(wlidPer%i''=%i)&(tankFlag''=%i);\n' , LECProbs(i), j, nextwlid,j+1);
            else
                fprintf(fid, ' %i:(wlidPer%i''=%i)&(tankFlag''=%i) +' , LECProbs(i), j, nextwlid,j+1);
            end
        end
    end
end


j=numTanks;

for k=1:wlidMax
    currcell=lattice(k);
    wlid=currcell.wlid;
    wl = currcell.wl;

    fprintf(fid, '    [] currN=0&sink=0&tankFlag=%i&wlid%i=%i -> ',j,j,wlid);
%     fprintf(fid, "%i:(wlidPer%i'=0)&(currN'=1)&(tankFlag'=1) + %i:(wlidPer%i'=wlidMax)&(currN'=1)&(tankFlag'=1) + ",LECLowerProb,j,LECUpperProb,j);
    for i=1:numBins
        nextwl = wl+i-(numBins+1)/2;
        nextwlid = ceil(nextwl/deltawl);
        nextwlid = max(0,nextwlid);
        nextwlid = min(wlidMax-1,nextwlid);
        
        
        if i==numBins
            fprintf(fid, ' %i:(wlidPer%i''=%i)&(currN''=1)&(tankFlag''=1);\n' , LECProbs(i), j, nextwlid);
        else
            fprintf(fid, ' %i:(wlidPer%i''=%i)&(currN''=1)&(tankFlag''=1) +' , LECProbs(i), j, nextwlid);
        end
    end
end
    
    
fprintf(fid, '\n \n');


end