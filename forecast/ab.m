%% absolute Bias
function ab = ab(y_obs,y_sim)
j=length(y_obs);
Diff=zeros(j,1);
for i=1:j
    Diff(i,1)=abs(y_obs(i,1)-y_sim(i,1));
end
ab=(sum(Diff))/j;
end


    
    