function[] = printControllerWaterTankMultiTank(lattice, maxN, fid,deltawl,numTanks)

wlContLowThresh=20;
wlContUpThresh=80;

wlContLowThreshid = ceil(wlContLowThresh/deltawl);
wlContUpThreshid = ceil(wlContUpThresh/deltawl);

for i=1:numTanks
%     if(deltawl==1)
%         fprintf(fid,'    [] currN=1&wlidPer%i<%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i)&(tankFlag''=%i);\n', i, wlContLowThreshid, i, i, i+1);
%         fprintf(fid,'    [] currN=1&contAction%i=0&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, i, wlContLowThreshid, i, i, i+1);
% 
%         fprintf(fid,'    [] currN=1&contAction%i!=0&wlidPer%i<%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, i, wlContUpThreshid, i, i, i+1);
%         fprintf(fid,'    [] currN=1&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, wlContUpThreshid, i, i, i+1);
    if(mod(wlContLowThresh,deltawl)  ==  1  &&  mod(wlContUpThresh,deltawl)  ==  0)
        fprintf(fid,'    [] currN=1&wlidPer%i<%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i)&(tankFlag''=%i);\n', i, wlContLowThreshid, i, i, i+1);
        fprintf(fid,'    [] currN=1&contAction%i=0&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, i, wlContLowThreshid, i, i, i+1);

        fprintf(fid,'    [] currN=1&contAction%i!=0&wlidPer%i<%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, i, wlContUpThreshid, i, i, i+1);
        fprintf(fid,'    [] currN=1&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, wlContUpThreshid, i, i, i+1);
    elseif(mod(wlContLowThresh,deltawl)  ==  1  &&  mod(wlContUpThresh,deltawl) ~=  0)
        fprintf(fid,'    [] currN=1&wlidPer%i<%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, wlContLowThreshid, i, i, i+1);
        fprintf(fid,'    [] currN=1&contAction%i=0&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, i, wlContLowThreshid, i, i, i+1);

        fprintf(fid,'    [] currN=1&contAction%i!=0&wlidPer%i<=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, i, wlContUpThreshid, i, i, i+1);
        fprintf(fid,'    [] currN=1&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, wlContUpThreshid, i, i, i+1);

    elseif(mod(wlContLowThresh,2) ~=  1  &&  mod(wlContUpThresh,2)  ==  0)
        fprintf(fid,'    [] currN=1&wlidPer%i<=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, wlContLowThreshid, i, i, i+1);
        fprintf(fid,'    [] currN=1&contAction%i=0&wlidPer%i>%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, i, wlContLowThreshid, i, i, i+1);

        fprintf(fid,'    [] currN=1&contAction%i!=0&wlidPer%i<%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, i, wlContUpThreshid, i, i, i+1);
        fprintf(fid,'    [] currN=1&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, wlContUpThreshid, i, i, i+1);

    else
        fprintf(fid,'    [] currN=1&wlidPer%i<=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, wlContLowThreshid, i, i+1);
        fprintf(fid,'    [] currN=1&contAction%i=0&wlidPer%i>%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, i, wlContLowThreshid, i, i+1);

        fprintf(fid,'    [] currN=1&contAction%i!=0&wlidPer%i<=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=1)&(tankFlag''=%i);\n', i, i, wlContUpThreshid, i, i+1);
        fprintf(fid,'    [] currN=1&wlidPer%i>=%i&sink=0&tankFlag=%i -> (currN''=1)&(contAction%i''=0)&(tankFlag''=%i);\n', i, wlContUpThreshid, i, i+1);

    end
    fprintf(fid,'\n\n');
end


fprintf(fid,'    [] currN=1&sink=0&tankFlag=%i&contAction1!=contAction2 -> (currN''=2)&(tankFlag''=1)&(contAction1G''=contAction1)&(contAction2G''=contAction2);\n',(numTanks+1));
fprintf(fid,'    [] currN=1&sink=0&tankFlag=%i&contAction1=0&contAction2=0 -> (currN''=2)&(tankFlag''=1)&(contAction1G''=contAction1)&(contAction2G''=contAction2);\n',(numTanks+1));
fprintf(fid,'    [] currN=1&sink=0&tankFlag=%i&contAction1=1&contAction2=1&wlidPer1<=wlidPer2 -> (currN''=2)&(tankFlag''=1)&(contAction1G''=contAction1)&(contAction2G''=0);\n',(numTanks+1));
fprintf(fid,'    [] currN=1&sink=0&tankFlag=%i&contAction1=1&contAction2=1&wlidPer2<wlidPer1 -> (currN''=2)&(tankFlag''=1)&(contAction2G''=contAction2)&(contAction1G''=0);\n',(numTanks+1));

fprintf(fid,'\n\n');


% add logic to pick control action

end