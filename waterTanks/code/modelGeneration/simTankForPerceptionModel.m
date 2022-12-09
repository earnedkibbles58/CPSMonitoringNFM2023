seed = 23443;
rng(seed);

numTrials = 1;

unsafes = 0;

allPerErrs = [];

for j=1:numTrials
    inflows = [13.5];
    outflows = [4.3];
    
    inflows_est = [12 13 14 15];
    outflows_est = [3 4 5 6];
    
%     inflows_est = [13.5];
%     outflows_est = [4.3];
    
    wlMax=100;
    wlInitLow = 40;
    wlInitHigh = 60;
    wlInit1=unifrnd(wlInitLow,wlInitHigh);
    wlInit2=unifrnd(wlInitLow,wlInitHigh);
    
    ctrlThreshLower = 20;
    ctrlThreshUpper = 80;
    
    numSteps = 300;%50;
    contAction1 = 0;
    contAction2 = 0;
    
    unsafe = 0;
    % noise params
    mu = 0;
    sigma = 5;
    noiseDist = makedist('Normal','mu',mu,'sigma',sigma);
    minValProb = 0.1;
    maxValProb = 0.1;

    
    wl1 = wlInit1;
    noises1 = [];
    wls1 = [];
    wlPers1 = [];
    wlEsts1 = [];
    wlEstErrs1 = [];
    
    wl2 = wlInit2;
    noises2 = [];
    wls2 = [];
    wlPers2 = [];
    wlEsts2 = [];
    wlEstErrs2 = [];

    % initialize bayesian filter
    filter_wl_disc = 1;
    stateDist1 = [];
    curr_wl = filter_wl_disc/2;
    while(1)
        if curr_wl>wlMax
            break
        end
        if curr_wl>wlInitLow && curr_wl<wlInitHigh
            stateDist1 = [stateDist1 1];
        else
            stateDist1 = [stateDist1 0];
        end
        curr_wl = curr_wl + filter_wl_disc;
    end
    stateDist1 = stateDist1/sum(stateDist1);
    
    % initialize bayesian filter
    stateDist2 = [];
    curr_wl = filter_wl_disc/2;
    while(1)
        if curr_wl>wlMax
            break
        end
        if curr_wl>wlInitLow && curr_wl<wlInitHigh
            stateDist2 = [stateDist2 1];
        else
            stateDist2 = [stateDist2 0];
        end
        curr_wl = curr_wl + filter_wl_disc;
    end
    stateDist2 = stateDist2/sum(stateDist2);

    for i=1:numSteps
        % run perception tank 1
        r=rand;
        if(r<minValProb)
            noise1=-100;
        elseif(r>(1-maxValProb))
            noise1=100;
        else
            noise1 = random(noiseDist,1);
        end
        noises1 = [noises1; noise1];
        %     noise = randn*sigma + mu;
        wlPer1 = max(min(wl1+noise1,wlMax),0);
        wlPers1 = [wlPers1;wlPer1];
        
        stateDist1 = bayesMonitorPerception(stateDist1,wlPer1,noiseDist,filter_wl_disc,minValProb,maxValProb,wlMax);
        wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc);
        
        
        % run perception tank 2
        r=rand;
        if(r<minValProb)
            noise2=-100;
        elseif(r>(1-maxValProb))
            noise2=100;
        else
            noise2 = random(noiseDist,1);
        end
        noises2 = [noises2; noise2];
        %     noise = randn*sigma + mu;
        wlPer2 = max(min(wl2+noise2,wlMax),0);
        wlPers2 = [wlPers2;wlPer2];
        
        stateDist2 = bayesMonitorPerception(stateDist2,wlPer2,noiseDist,filter_wl_disc,minValProb,maxValProb,wlMax);
        wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc);

        % compute control tank 1
        if wlEst1<ctrlThreshLower || (wlEst1<ctrlThreshUpper && contAction1==1)
            contAction1=1;
        else
            contAction1=0;
        end
        
        % compute control tank 2
        if wlEst2<ctrlThreshLower || (wlEst1<ctrlThreshUpper && contAction2==1)
            contAction2=1;
        else
            contAction2=0;
        end

        
        
        %% Global controller
        contActionG1=contAction1;
        contActionG2=contAction2;
        if(contAction1==1 && contAction2==1 && wlPer1<=wlPer2)
            contActionG1=1;
            contActionG2=0;
        elseif(contAction1==1 && contAction2==1 && wlPer1>wlPer2)
            contActionG1=0;
            contActionG2=1;
        elseif(contAction1==1 && contAction2==1)
            r = rand;
            if r<=0.5
                contActionG1=1;
                contActionG2=0;
            else
                contActionG1=0;
                contActionG2=1;

            end
        end
        
        
        
        %% Update water levels
        %     contAction
        % simulate tanks
        wl1=wl1-randsample(outflows,1)+contActionG1*randsample(inflows,1);
        if(wl1<=0 || wl1>wlMax)
            unsafe=1;
            break
        end
        wls1 = [wls1;wl1];

         wl2=wl2-randsample(outflows,1)+contActionG2*randsample(inflows,1);
        if(wl2<=0 || wl2>wlMax)
            unsafe=1;
            break
        end
        wls2 = [wls2;wl2];

        % update filter
        stateDist1 = bayesMonitorDynamics(stateDist1,contActionG1,inflows_est,outflows_est,filter_wl_disc,wlMax);
        wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc);
        wlEsts1 = [wlEsts1;wlEst1];
        wlEstErrs1 = [wlEstErrs1 wlEst1-wl1];
        allPerErrs = [allPerErrs wlEst1-wl1];

        stateDist2 = bayesMonitorDynamics(stateDist2,contActionG2,inflows_est,outflows_est,filter_wl_disc,wlMax);
        wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc);
        wlEsts2 = [wlEsts2;wlEst2];
        wlEstErrs2 = [wlEstErrs2 wlEst2-wl2];
        allPerErrs = [allPerErrs wlEst2-wl2];

    end
    
    
%     figure(1)
%     clf
%     plot(wls1,'b-')
%     hold on
%     plot(wlEsts1,'r-')
%     legend("Water Level","Estimate")
% 
%     figure(2)
%     clf
%     plot(wls2,'b-')
%     hold on
%     plot(wlEsts2,'r-')
%     legend("Water Level","Estimate")
% 
%     figure(3)
%     plot(wlEstErrs1)
% 
%     figure(4)
%     plot(wlEstErrs2)
    %     unsafe
    unsafes=unsafes+unsafe;
    
%     if(j==numTrials)
%         figure(1)
%         plot(noises1)
% 
%         figure(2)
%         plot(wls1)
%         hold on
%         plot(wls2)
%         hold off
% 
%         figure(3)
%         plot(wlPers1)
%         hold on
%         plot(wlPers2)
%         yline(20)
%         yline(80)
%         hold off
%     end
    
end
unsafes
unsafes/numTrials


maxPerErr = max(abs(allPerErrs));

figure(5)
clf
histogram(allPerErrs,-maxPerErr:filter_wl_disc:maxPerErr)


% print perception model and save data
err_dict = containers.Map(0,0);

for i=1:length(allPerErrs)
    bin_ind = round(allPerErrs(i));
    if isKey(err_dict,bin_ind)
        err_dict(bin_ind) = err_dict(bin_ind) + 1/length(allPerErrs);
    else
        err_dict(bin_ind) = 1/length(allPerErrs);
    end
    
end


thisKeys = err_dict.keys;
for i=1:length(thisKeys)
    thisKey = thisKeys(i);
    thisValue = err_dict(thisKey{1});
    [thisKey{1} thisValue]
    
    
end



function[wlEst] = wlEstFromStateDist(stateDist,wlDisc)

    wlEst = 0;
    for i=1:length(stateDist)
        curr_wl_prob = stateDist(i);
        curr_wl = wlDisc*(i-1)+0.5;
        wlEst = wlEst + curr_wl_prob*curr_wl;
    end
end


function[stateDist] = bayesMonitorUpdate(stateDist,controlCommand,inflows,outflows,wlReading,wlDisc,wlMax,noiseDist,minValProb,maxValProb)

    stateDist = bayesMonitorPerception(stateDist,wlReading,noiseDist,wlDisc,minValProb,maxValProb);
    stateDist = bayesMonitorDynamics(stateDist,controlCommand,inflows,outflows,wlDisc,wlMax);
    
end


function[stateDist] = bayesMonitorDynamics(stateDist,controlCommand,inflows,outflows,wlDisc,wlMax)

    newStateDist = zeros(size(stateDist));
    
    for i=1:length(stateDist)
        curr_wl_prob = stateDist(i);
        curr_wl = wlDisc*(i-1)+0.5;
        
        for j=1:length(inflows)
            
            for k=1:length(outflows)
                next_wl = curr_wl+inflows(j)*controlCommand-outflows(k);
                next_wl = max(next_wl,0);
                next_wl = min(next_wl,wlMax);
                bin_ind = min(getBin(next_wl,wlDisc),length(stateDist));
                newStateDist(bin_ind) = newStateDist(bin_ind) + curr_wl_prob*(1/length(inflows))*(1/length(outflows));
            end
        end
        
    end
    
    stateDist = newStateDist;

end


function[stateDist] = bayesMonitorPerception(stateDist,wlReading,noiseDist,wlDisc,minValProb,maxValProb,wlMax)

    
    newStateDist = zeros(size(stateDist));
    for i=1:length(stateDist)
        curr_wl_prob = stateDist(i);
        curr_wl = wlDisc*(i-1)+0.5;
        prob = probOfReading(curr_wl,wlReading,noiseDist,minValProb,maxValProb,wlMax);
        newStateDist(i) = curr_wl_prob*prob;
        
    end
    stateDist = newStateDist/sum(newStateDist);
end


function[bin] = getBin(wl,wlDisc)

    bin = wl-mod(wl,wlDisc)+1;
end

function[probability] = probOfReading(wl,wlReading,noiseDist,minValProb,maxValProb,wlMax)
    
    probability = 0;
    if wlReading == 0
        probability = minValProb;
    elseif wlReading == wlMax
        probability = maxValProb;
    end
    
    probability = probability + (1-minValProb-maxValProb)*pdf(noiseDist,wlReading-wl);
    
    
end