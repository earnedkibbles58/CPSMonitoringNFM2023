function []=convertOlegBaselineToPRISMModelMultiTank(lattice, maxN, fileName,deltawl,trimmed,numTanks)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

format long

fid=fopen(fileName, 'wt');

latticeSize=size(lattice);

wlidMax=latticeSize(2);

fprintf(fid, 'mdp\n');
fprintf(fid, '\n \n');
for i=1:numTanks
    fprintf(fid, 'const int wlidInit%i;\n',i);
    fprintf(fid, 'const int initContAction%i;\n',i);
end

fprintf(fid, 'const int wlidMax=%i;\n', wlidMax-1);
fprintf(fid, 'const int Nmax=%i;\n', maxN);
% fprintf(fid, 'const int wlContLowThresh=15;\n');
% fprintf(fid, 'const int wlContUpThresh=45;\n');
fprintf(fid, 'const int maxTime=10;\n');

fprintf(fid, '\n \n');
fprintf(fid, 'module PRISMLattice\n\n');
for i=1:numTanks
    fprintf(fid, '    wlid%i : [0..wlidMax] init wlidInit%i;\n',i,i);
    fprintf(fid, '    wlidPer%i : [0..wlidMax] init wlidInit%i;\n',i,i);
    fprintf(fid, '    contAction%i : [0..1] init initContAction%i;\n',i,i);
    fprintf(fid, '    contAction%iG : [0..1] init initContAction%i;\n',i,i);
    fprintf(fid, '\n');
end
fprintf(fid, '    currN : [0..2] init 0;\n');
fprintf(fid, '    tankFlag : [1..%i] init 1;\n', numTanks+1);
fprintf(fid, '    sink: [0..1] init 0;\n');
fprintf(fid, '    timesteps: [0..maxTime] init 0;\n');

printLECModelWaterTankMultiTankSimDataNew(lattice, maxN, fid,deltawl,numTanks)

printControllerWaterTankMultiTank(lattice, maxN, fid,deltawl,numTanks)

for j=1:numTanks
    for i=1:wlidMax
        currcell=lattice(i);
        currwlid=currcell.wlid;

%         if(deltawl == 1)
%             currwl=(currwlid+1)*deltawl;
%         elseif(deltawl == 2)
%             currwl=(currwlid)*deltawl;
%         end
        currTransMap=currcell.nextPoints;
        currMapKeys=keys(currTransMap);
        for k=1:length(currMapKeys)
            key=currMapKeys{k};
            if key  ==  0
                continue
            end
            currN=key;
            fprintf(fid, '\n');
            val=currTransMap(key);

            for l=1:length(val)
                currK=l-1;
                nextPointsTrans=val{l};
                for m=1:size(nextPointsTrans,1)
                    nextPointwlid=nextPointsTrans(m,1);
                    if(nextPointwlid<= 0 || currwlid<= 0)
                        fprintf(fid, '    [] wlid%i=%i&currN=%i&contAction%iG=%i&sink=0&timesteps<maxTime&tankFlag=%i -> (sink''=1);\n', j, currwlid, 2, j, currK, j);
                    elseif(nextPointwlid>= wlidMax-1 || currwlid>= wlidMax-1)
                        fprintf(fid, '    [] wlid%i=%i&currN=%i&contAction%iG=%i&sink=0&timesteps<maxTime&tankFlag=%i -> (sink''=1);\n', j, currwlid, 2, j, currK, j);
                    else
                        if(j == numTanks)
                            fprintf(fid, '    [] wlid%i=%i&currN=%i&contAction%iG=%i&sink=0&timesteps<maxTime&tankFlag=%i -> (wlid%i''=%i)&(currN''=0)&(timesteps''=timesteps+1)&(tankFlag''=%i);\n', j, currwlid, 2, j, currK, j, j, round(nextPointwlid), 1);
                        else
                            fprintf(fid, '    [] wlid%i=%i&currN=%i&contAction%iG=%i&sink=0&timesteps<maxTime&tankFlag=%i -> (wlid%i''=%i)&(tankFlag''=%i);\n', j, currwlid, 2, j, currK, j, j, round(nextPointwlid), j+1);
                        end
                    end
                end
            end
        end
    end
end
fprintf(fid, '\n \n');
fprintf(fid, 'endmodule\n');
fclose(fid);
end

