function[] = printLECModelWaterTankMultiTankNew(lattice, maxN, fid,deltawl,numTanks)


latticeSize=size(lattice);
wlidMax=latticeSize(2);

fprintf(fid, '\n \n');


% LECProbs = [0.020500000000000   0.024900000000000   0.036800000000000   0.053500000000000   0.076400000000000   0.095200000000000   0.102800000000000   0.093200000000000   0.073300000000000   0.052700000000000   0.040700000000000   0.023000000000000   0.020200000000000];
% LECLowerProb = 0.143200000000000;
% LECUpperProb = 0.143600000000000;

%            -6      -5     -4     -3     -2    -1      0       1     2       3      4     5
% LECProbs = [0.0017 0.0267 0.0450 0.1083 0.1800 0.2017 0.1983 0.1217 0.0683 0.0267 0.0183 0.0033];


% LECProbs = [0.000402333534500 0.000201166767250 0.000603500301750 0.001307583987125 0.002514584590626 0.011265338966003 0.022832428082881 0.054616777308388 0.113960973647151 ...
%     0.173606920136796 0.195433514383428 0.179742506537923 0.119995976664652 0.070609535304766 0.031482599074633 0.011667672500503 0.005029169181251 0.002112251056126 ...
%     0.000704083685375 0.001005833836250 0.000402333534500 0.000301750150875 0.000201166767250];

% deltawl=5
% if deltawl==5
%     % deltawl=5
% %     LECProbs = [0.1817 0.77 0.0483];
%     
%     LECProbs = [0.000402333534500+0.000201166767250+0.000603500301750+0.001307583987125 0.002514584590626+0.011265338966003+0.022832428082881+0.054616777308388+0.113960973647151 ...
%     0.173606920136796+0.195433514383428+0.179742506537923+0.119995976664652+0.070609535304766 0.031482599074633+0.011667672500503+0.005029169181251+0.002112251056126+0.000704083685375 ...
%     0.001005833836250+0.000402333534500+0.000301750150875+0.000201166767250];
% elseif deltawl==2
%     % deltawl=2
% %     LECProbs = [(0.0017+0.0267) (0.0450+0.1083) (0.1800+0.2017) (0.1983+0.1217) (0.0683+0.0267) (0.0183+0.0033)];
%     
%     LECProbs = [0.000402333534500+0.000201166767250 0.000603500301750+0.001307583987125 0.002514584590626+0.011265338966003 0.022832428082881+0.054616777308388 ...
%         0.113960973647151+0.173606920136796 0.195433514383428+0.179742506537923 0.119995976664652+0.070609535304766 0.031482599074633+0.011667672500503 0.005029169181251+0.002112251056126 ...
%     0.000704083685375+0.001005833836250 0.000402333534500+0.000301750150875 0.000201166767250];
% 
% elseif deltawl==1
% %     LECProbs = [0.0017 0.0267 0.0450 0.1083 0.1800 0.2017 0.1983 0.1217 0.0683 0.0267 0.0183 0.0033];
%     
%     LECProbs = [0.000402333534500 0.000201166767250 0.000603500301750 0.001307583987125 0.002514584590626 0.011265338966003 0.022832428082881 0.054616777308388 0.113960973647151 ...
%     0.173606920136796 0.195433514383428 0.179742506537923 0.119995976664652 0.070609535304766 0.031482599074633 0.011667672500503 0.005029169181251 0.002112251056126 ...
%     0.000704083685375 0.001005833836250 0.000402333534500 0.000301750150875 0.000201166767250];
% else
%     error('LEC model not hardcoded for deltawl=%i',deltawl)
% end

% old
% LECProbs = [0.000402333534500 0.000201166767250 0.000603500301750 0.001307583987125 0.002514584590626 0.011265338966003 0.022832428082881 0.054616777308388 0.113960973647151 ...
%     0.173606920136796 0.195433514383428 0.179742506537923 0.119995976664652 0.070609535304766 0.031482599074633 0.011667672500503 0.005029169181251 0.002112251056126 ...
%     0.000704083685375 0.001005833836250 0.000402333534500 0.000301750150875 0.000201166767250];
% err_amnts = -11:1:11;


% updated
LECProbs = [0.0005 0.0011 0.0018 0.0058 0.0164 0.0316 0.0626 0.1126 0.1673 0.1895 0.1717 0.1193 0.0665 0.0287 0.0115 0.0061 0.0029 0.0019 0.0014 0.0006];
err_amnts = -10:1:9;
%% compute error probs for diff bins:
% create dict mapping error amount to probability
per_err_dict_interval = containers.Map(0,0);

for per_err_bin=1:length(LECProbs)
    err_amnt = err_amnts(per_err_bin);
    % map true err to interval err
    if err_amnt == 0
        bin_ind = 0;
    elseif err_amnt > 0
        bin_ind = ceil(err_amnt/deltawl);
    else
        bin_ind = floor(err_amnt/deltawl);
    end
        
    if isKey(per_err_dict_interval,bin_ind)
        per_err_dict_interval(bin_ind) = per_err_dict_interval(bin_ind) + LECProbs(per_err_bin);
    else
        per_err_dict_interval(bin_ind) = LECProbs(per_err_bin);
    end
end

% sum(LECProbs)
% 1-sum(LECProbs)

if abs(sum(LECProbs)-1) <= 0.01
    LECProbs = LECProbs/sum(LECProbs);
end

assert(abs(sum(LECProbs)-1)<= 0.0001,'LEC error probabilities do not sum to 1');

for j=1:numTanks-1
    
    fprintf(fid, '    [] currN=0&sink=0&tankFlag=%i -> ',j);
    map_keys = keys(per_err_dict_interval);
    for k=1:length(map_keys)
        this_key = map_keys{k};
        this_val = per_err_dict_interval(this_key);
        
        if k ==length(map_keys)
            fprintf(fid, ' %i:(wlidPer%i''=max(0,min(wlidMax,wlid%i+%i)))&(tankFlag''=%i);\n' , this_val, j,j,this_key,j+1);
        else
            fprintf(fid, ' %i:(wlidPer%i''=max(0,min(wlidMax,wlid%i+%i)))&(tankFlag''=%i) +' , this_val, j,j,this_key,j+1);
        end
    end
    
end

j=numTanks;


fprintf(fid, '    [] currN=0&sink=0&tankFlag=%i -> ',j);
map_keys = keys(per_err_dict_interval);
for k=1:length(map_keys)
    this_key = map_keys{k};
    this_val = per_err_dict_interval(this_key);

    if k ==length(map_keys)
        fprintf(fid, ' %i:(wlidPer%i''=max(0,min(wlidMax,wlid%i+%i)))&(currN''=1)&(tankFlag''=1);\n' , this_val, j,j,this_key);
    else
        fprintf(fid, ' %i:(wlidPer%i''=max(0,min(wlidMax,wlid%i+%i)))&(currN''=1)&(tankFlag''=1) +' , this_val, j,j,this_key);
    end
end
    
    
fprintf(fid, '\n \n');


end