# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:22:31 2022

@author: kgavahi
"""

import numpy as np


def hymod(xx1, xx2, xx3, xx4, xx5, Pars, I, Pet): 
    xt = np.zeros([5])
    zt = np.zeros([5])
    
    # HyMOD Parameters
    a = Pars[0];        # percentage quickflow
    b = Pars[1];        # shape of Pareto distribution
    cmax = Pars[2];     # maximum storage capacity 
    rs = Pars[3];       # slowflow time constant
    rq = Pars[4];       # quickflow time constant
    
    xt[0]=xx1
    xt[1]=xx2
    xt[2]=xx3
    xt[3]=xx4
    xt[4]=xx5
    s = xx1
    
    smax = cmax/(1+b)  # Maximum STorage Content
    if (s > smax): s=0.999*smax
    
    c_prev = cmax*(1-np.power((1-((b+1)*s/cmax)),(1/(b+1))))
    ER1 = max(I + c_prev - cmax,0.0)   # effective rainfall part 1 
    II = I-ER1
    dummy = min(((c_prev+II)/cmax),1)
    s1 = (cmax/(b+1))*(1-np.power((1-dummy),(b+1)))  # new state
    ER2 = max(II-(s1-s),0)     # effective rainfall part 2 
    evap = s1/smax*Pet        # actual ET is linearly related to the 
                               # soil moisture state 
    s1 = max(s1-evap, 0)      # update state
    UQ = ER1+a*ER2            # quickflow contribution
    US = (1-a)*ER2            # slowflow contribution
    
    inp=UQ
    xt[0]=s1                  # Nonlinear Storage Content
    n=1
    
    
    while n < 4:
        xt[n] = (1-rq)*xt[n]+(1-rq)*inp   # Forecast Step
        zt[n]= (rq/(1-rq))*xt[n]		  # Generated observation	
        inp = zt[n]       
        n=n+1
    
    xt[n] = (1-rs)*xt[n]+(1-rs)*US       # Forecast Step
    zt[n] = (rs/(1-rs))*xt[n]           # Generated observation   
    qt=zt[3]+zt[4]			              # Generated total observation
    
    xx1=xt[0]
    xx2=xt[1]
    xx3=xt[2]
    xx4=xt[3]
    xx5=xt[4]
    
    #xx1: nolinear tank (soil moisture storage)
    #xx2: quick flow tank storage 
    #xx3: quick flow tank storage
    #xx4: quick flow tank storage
    #xx5: slow flow tank storage
    return xx1,xx2,xx3,xx4,xx5,qt



