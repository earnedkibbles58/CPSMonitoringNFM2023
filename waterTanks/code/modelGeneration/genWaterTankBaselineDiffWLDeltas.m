%% This script builds the lattice
% The lattice is a 1-d array of custom structs
% Each struct contains the water level (and unique id), along with a map struct that maps 
% Ns to an array which corresonds to the lattice points that the K value
% associated with the array index maps to

clear; close all;
format long

%% Define hyper parameters
inflow = 13.5;
outflow = 4.3;
wlMax=100;
N=1;
trimmed=0;

% deltawls = [1 2 3 4 5 6 7 8 9 10];
deltawls = [5];
for i=1:length(deltawls)
    
    deltawl = deltawls(i);
    %% Untrimmed model
    baseline = initWaterTankBaseline(wlMax,deltawl);
    lattice = addWaterTankBaselineTransitions(baseline,N,deltawl,inflow,outflow,trimmed);
    modelFolder = '../../models/noTrimming/if' + string(inflow) + '_of' + string(outflow) + '_deltawl' + string(deltawl);
    mkdir(modelFolder)
    modelFile = '../../models/noTrimming/if' + string(inflow) + '_of' + string(outflow) + '_deltawl' + string(deltawl) + '/waterTankBaseline.prism';
    convertWaterTankBaselineToPRISMModelMultiTank(lattice,N,modelFile,deltawl,trimmed,2);
end