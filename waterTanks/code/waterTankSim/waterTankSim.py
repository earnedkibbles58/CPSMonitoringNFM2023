



import numpy as np
import random
import scipy.stats




def main():

    numTrials = 100
    unsafes = 0
    allPerErrs = []

    for j in range(numTrials):
        inflows = [13.5]
        outflows = [4.3]
        
        inflows_est = [12 13 14 15]
        outflows_est = [3 4 5 6]
        
        
        wlMax=100
        wlInitLow = 40
        wlInitHigh = 60
        wlInit1=random.uniform(wlInitLow,wlInitHigh)
        wlInit2=random.uniform(wlInitLow,wlInitHigh)
                
        ctrlThreshLower = 20
        ctrlThreshUpper = 80
        
        numSteps = 50
        contAction1 = 0
        contAction2 = 0
        
        unsafe = 0

        mu = 0
        sigma = 5
        # noiseDist = makedist('Normal','mu',mu,'sigma',sigma)
        np.random.normal(mu,sigma,1)[0] ## SEEME: command to generate samples from noise distribution
        # noiseDist = scipy.stats.norm(mu,sigma)
        minValProb = 0.1
        maxValProb = 0.1

        
        wl1 = wlInit1
        noises1 = []
        wls1 = []
        wlPers1 = []
        wlEsts1 = []
        wlEstErrs1 = []
        
        wl2 = wlInit2
        noises2 = []
        wls2 = []
        wlPers2 = []
        wlEsts2 = []
        wlEstErrs2 = []

        filter_wl_disc = 1
        stateDist1 = []
        curr_wl = filter_wl_disc/2
        while True:
            if curr_wl>wlMax:
                break
            
            if curr_wl>wlInitLow and curr_wl<wlInitHigh:
                stateDist1.append(1)
            else
                stateDist1.append(0)
            
            curr_wl = curr_wl + filter_wl_disc
        
        stateDist1 = stateDist1/sum(stateDist1)
        
        stateDist2 = []
        curr_wl = filter_wl_disc/2
        while True
            if curr_wl>wlMax:
                break
            
            if curr_wl>wlInitLow and curr_wl<wlInitHigh:
                stateDist2.append(1)
            else
                stateDist2.append(0)
            
            curr_wl = curr_wl + filter_wl_disc
        
        stateDist2 = stateDist2/sum(stateDist2)

        for i in range(numSteps):
            r=rand
            if(r<minValProb):
                noise1=-100
            elif(r>(1-maxValProb)):
                noise1=100
            else:
                noise1 = np.random.normal(mu,sigma,1)[0]
            
            noises1.append(noise1)
            wlPer1 = max(min(wl1+noise1,wlMax),0)
            wlPers1.append(wlPer1)
            
            stateDist1 = bayesMonitorPerception(stateDist1,wlPer1,mu,sigma,filter_wl_disc,minValProb,maxValProb,wlMax)
            wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
            
            
            r=rand
            if(r<minValProb):
                noise2=-100
            elif(r>(1-maxValProb)):
                noise2=100
            else
                noise2 = np.random.normal(mu,sigma,1)[0]
            
            noises2.append(noise2)
            wlPer2 = max(min(wl2+noise2,wlMax),0)
            wlPers2.append(wlPer2)
            
            stateDist2 = bayesMonitorPerception(stateDist2,wlPer2,mu,sigma,filter_wl_disc,minValProb,maxValProb,wlMax)
            wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)

            ## compute control tank 1
            if wlEst1<ctrlThreshLower or (wlEst1<ctrlThreshUpper and contAction1==1):
                contAction1=1
            else:
                contAction1=0
            
            
            ## compute control tank 2
            if wlEst2<ctrlThreshLower or (wlEst1<ctrlThreshUpper and contAction2==1):
                contAction2=1
            else:
                contAction2=0
            

            
            
            ## Global controller
            contActionG1=contAction1
            contActionG2=contAction2
            if(contAction1==1 and contAction2==1 and wlPer1<wlPer2):
                contActionG1=1
                contActionG2=0
            elif(contAction1==1 and contAction2==1 and wlPer1>wlPer2):
                contActionG1=0
                contActionG2=1
            elif(contAction1==1 and contAction2==1 and wlPer1 == wlPer2):
                r = random.uniform(0,1)
                if r<=0.5:
                    contActionG1=1
                    contActionG2=0
                else:
                    contActionG1=0
                    contActionG2=1
            
            
            
            wl1=wl1-random.choice(outflows)+contActionG1*random.choice(inflows)
            if(wl1<=0 or wl1>wlMax):
                unsafe=1
                break
            
            wls1.append(wl1)

            wl2=wl2-random.choice(outflows)+contActionG2*random.choice(inflows)
            if(wl2<=0 or wl2>wlMax):
                unsafe=1
                break
            
            wls2.append(wl2)

            ## update filter
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
        
        
        unsafes=unsafes+unsafe;
        
        
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





def wlEstFromStateDist(stateDist,wlDisc):

    wlEst = 0
    for i in range(len(stateDist)):
        curr_wl_prob = stateDist(i)
        curr_wl = wlDisc*(i-1)+0.5
        wlEst = wlEst + curr_wl_prob*curr_wl
    
    return wlEst



function[stateDist] = bayesMonitorUpdate(stateDist,controlCommand,inflows,outflows,wlReading,wlDisc,wlMax,noiseDist,minValProb,maxValProb)

    stateDist = bayesMonitorPerception(stateDist,wlReading,mu,sigma,wlDisc,minValProb,maxValProb);
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


def bayesMonitorPerception(stateDist,wlReading,mu,sigma,wlDisc,minValProb,maxValProb,wlMax):

    
    newStateDist = np.zeros(len(stateDist))
    for i in range(len(stateDist)):
        curr_wl_prob = stateDist(i)
        curr_wl = wlDisc*(i-1)+0.5
        prob = probOfReading(curr_wl,wlReading,mu,sigma,minValProb,maxValProb,wlMax)
        newStateDist[i] = curr_wl_prob*prob
        
    
    stateDist = newStateDist/sum(newStateDist);
    return stateDist



function[bin] = getBin(wl,wlDisc)

    bin = wl-mod(wl,wlDisc)+1;
end

def probOfReading(wl,wlReading,mu,sigma,minValProb,maxValProb,wlMax):
    
    probability = 0
    if wlReading == 0:
        probability = minValProb
    elif wlReading == wlMax:
        probability = maxValProb

    noiseDist = scipy.stats.norm(mu,sigma)
    
    probability = probability + (1-minValProb-maxValProb)*noiseDist.pdf(wlReading-wl);
    return probability





if __name__ == '__main__':
    main()
