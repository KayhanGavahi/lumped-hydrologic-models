function  kge = KGE(y_sim, y_obs)

sd_y_sim=std(y_sim);
sd_y_obs=std(y_obs);
 
avr_y_sim=mean(y_sim);
avr_y_obs=mean(y_obs);

r=corr(y_sim, y_obs);
alpha=sd_y_sim/sd_y_obs;
beta=avr_y_sim/avr_y_obs;

kge = 1- sqrt( ((r-1)^2) + ((alpha-1)^2)  + ((beta-1)^2) );
 
end