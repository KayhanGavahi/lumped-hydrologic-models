# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:38:46 2022

@author: kgavahi
"""
import numpy as np



def fland1(ep, pxv, par, states):
    eps = 2.22044604925031e-16 # from Matlab eps
#   THIS SUBROUTINE EXECUTES THE SAC-SMA OPERATION FOR ONE
#   TIME PERIOD.    
    TIMESPERDAY=4
    
    uztwc = states[1-1]; uzfwc = states[2-1]; lztwc = states[3-1] # -1 for python
    lzfsc = states[4-1]; lzfpc = states[5-1]; adimc = states[6-1] 
    xquick1=states[7-1]; xquick2=states[8-1]; xquick3=states[9-1]
    
    uztwm = par[1-1];uzfwm =par[2-1];uzk= par[3-1];pctim = par[4-1];adimp =par[5-1] # -1 for python
    zperc =par[6-1];rexp= par[7-1];lztwm =par[8-1];lzfsm =par[9-1];lzfpm=par[10-1]
    lzsk= par[11-1];lzpk= par[12-1];pfree=par[13-1]; Kq=par[14-1]
    riva  = par[15-1];rserv = par[16-1];side = par[17-1];pxmlt = 1;zp = zperc
    
    # COMPUTE EVAPOTRANSPIRATION LOSS FOR THE TIME INTERVAL
    # EDMND IS THE ET-DEMAND FOR THE TIME INTERVAL
    edmnd = ep;
    e1 = edmnd * uztwc/uztwm
    red = edmnd - e1 # red is residual evap demand
    uztwc = uztwc-e1
    if np.abs(uztwc) < eps: uztwc = 0
    e2 = 0
    index = 0     # This is an index to get rid of "goto L225" command existed in C code
    if uztwc < 0:   # e1 cannot exceed uztwc
        e1 = e1 + uztwc;
        uztwc = 0;
        red = edmnd - e1;
        if uzfwc < red:
            e2 = uzfwc;   # e2 is evap from uzfwc
            uzfwc = 0;
            red = red - e2;
            index = 1;   
        else:
            e2 = red;
            uzfwc = uzfwc - e2;
            red = 0;
    # UPPER ZONE FREE WATER RATIO EXCEEDS UPPER ZONE TENSION
    # WATER RATIO, THUS TRANSFER FREE WATER TO TENSION
    if index == 0:
        if (uztwc/uztwm) < (uzfwc/uzfwm):
            uzrat = (uztwc + uzfwc) / (uztwm + uzfwm)
            uztwc = uztwm * uzrat
            uzfwc = uzfwm * uzrat
    
    e3 = red * lztwc/(uztwm +lztwm)
    lztwc = lztwc - e3
    if np.abs(lztwc) < eps: lztwc = 0
    if lztwc < 0:
        e3 = e3 + lztwc # e3 cannot exceed lztwc
        lztwc = 0
    
    ratlzt = lztwc/lztwm
    saved = rserv * (lzfpm + lzfsm)
    ratlz = (lztwc + lzfpc + lzfsc - saved)/(lztwm + lzfpm + lzfsm - saved)
    
    if ratlzt < ratlz:     # RESUPPLY LOWER ZONE TENSION WATER FROM LOWER ZONE FREE WATER
                           # IF MORE WATER AVAILABLE THERE.
        delt = (ratlz - ratlzt) * lztwm
        lztwc = lztwc + delt
        lzfsc = lzfsc - delt
        if lzfsc < 0:      # IF TRANSFER EXCEEDS LZFSC THEN REMAINDER COMES FROM LZFPC
            lzfpc =  lzfpc + lzfsc
            lzfsc = 0

    #   COMPUTE ET FROM ADIMP AREA.-E5 
    e5 = e1 + (red+e2) * (adimc - e1 - uztwc) / (uztwm + lztwm)
    #   ADJUST ADIMC,ADDITIONAL IMPERVIOUS AREA STORAGE, FOR EVAPORATION 
    adimc =  adimc - e5
    if np.abs(adimc) < eps: adimc = 0
    if adimc < 0:  # E5 CANNOT EXCEED ADIMC. 
        e5 = e5 + adimc
        adimc = 0

    e5 = e5 * adimp

    #    E5 IS ET FROM THE AREA ADIMP.
    #    COMPUTE PERCOLATION AND RUNOFF AMOUNTS. 
    
    twx = pxv + uztwc - uztwm

    if twx < 0:  # ALL MOISTURE HELD IN UZTW--NO EXCESS.  
        uztwc = uztwc + pxv
        twx = 0.
    else:  # MOISTURE AVAILABLE IN EXCESS OF UZTW STORAGE 
        uztwc = uztwm
        
    adimc = adimc + pxv - twx    
    
    #    COMPUTE IMPERVIOUS AREA RUNOFF. */
    roimp = pxv * pctim;
    
    #    SET SOME INITIAL VALUES TO ZERO */
    srot  = 0;simpvt = 0;srodt = 0;srost = 0;sintft = 0;sgwfp = 0;
    sgwfs = 0;srecht = 0;sett  = 0;se1   = 0;se3    = 0;se4   = 0;se5 = 0;
    #    ROIMP IS RUNOFF FROM THE MINIMUM IMPERVIOUS AREA.  
    simpvt = simpvt + roimp;
    
    #    INITIALIZE TIME INTERVAL SUMS. 
    sbf = 0; ssur = 0;sif = 0;sperc = 0;sdro = 0;spbf = 0;
    
    # DT IS THE LENGTH OF EACH TIME INTERVAL IN DAYS
    # DT IS USED TO CALCULATE dinc IN fland1
    # dt = 0.25;
    dt = 1/TIMESPERDAY ;
    #    DETERMINE COMPUTATIONAL TIME INCREMENTS FOR THE BASIC TIME INTERVAL
    #    NINC = NUMBER OF TIME INCREMENTS THAT THE TIME INTERVAL
    #    IS DIVIDED INTO FOR FURTHER SOIL-MOISTURE ACCOUNTING.
    #    NO ONE INCREMENT WILL EXCEED 5.0 MILLIMETERS OF UZFWC+PAV
    #    DINC = LENGTH OF EACH INCREMENT IN DAYS.
    #    PINC = AMOUNT OF AVAILABLE MOISTURE FOR EACH INCREMENT.
    #    ninc = (int) (1.0 + 0.2 * sma->uzfwc + twx);      
    #    ninc = int8(1.0 + 0.2 * (uzfwc + twx)+0.5);
    ninc = 1.0; 
    dinc = 1.0 / np.double(ninc) * dt;
    pinc = twx / np.double(ninc);
    
    #    COMPUTE FREE WATER DEPLETION FRACTIONS FOR
    #    THE TIME INCREMENT BEING USED-BASIC DEPLETIONS
    #    ARE FOR ONE DAY */
        
    duz = 1 - np.power((1 - uzk), dinc);
    dlzp = np.double (1 - np.power((1 - lzpk), dinc));
    dlzs = 1 - np.power((1 - lzsk), dinc);
    parea = 1 - adimp - pctim;

    #    START INCREMENTAL FOR LOOP FOR THE TIME INTERVAL.
    adsur = 0;
    #        COMPUTE DIRECT RUNOFF (FROM ADIMP AREA).
    #        ADDRO IS THE AMOUNT OF DIRECT RUNOFF FROM
    #        THE AREA ADIMP-SDRO IS THE SIX HOUR SUMMATION
    ratio = (adimc - uztwc) / lztwm;
    addro = pinc * (ratio * ratio);
    sdro = sdro + (addro * adimp);        
    
    #        COMPUTE BASEFLOW AND KEEP TRACK OF TIME INTERVAL SUM.
    bf = lzfpc * dlzp;
    lzfpc = lzfpc - bf;
    if lzfpc <= 1.e-4:
        bf = bf + lzfpc;
        lzfpc = 0;
    
    sbf = sbf + bf;
    spbf = spbf + bf;
    bf = lzfsc * dlzs;
    lzfsc = lzfsc - bf;
    
    if lzfsc <= 1.e-4:
        bf = bf + lzfsc;
        lzfsc = 0;
    
    sbf = sbf + bf;
    
    #     COMPUTE PERCOLATION-IF NO WATER AVAILABLE THEN SKIP */
    if (pinc + uzfwc) <= 1.e-2:
        uzfwc = uzfwc + pinc;
        adimc = adimc + pinc - addro - adsur;
        if adimc > (uztwm + lztwm):
            addro = addro + adimc - uztwm + lztwm;
            adimc = uztwm + lztwm;
        
    else:
        percm = lzfpm * dlzp + lzfsm * dlzs;
    
        if (zp < 0.0): zp = 0.0
        perc = percm * uzfwc / uzfwm;
        #        DEFR IS THE LOWER ZONE MOISTURE DEFICIENCY RATIO */
        defr = 1.0 - (lztwc+lzfpc+lzfsc)/(lztwm+lzfpm+lzfsm);
       
        if defr < 0.0:
            defr;
            #[lztwc;lzfpc;lzfsc];
            #[lztwm;lzfpm;lzfsm];
        
        uzdefr = 1.0 - (uztwc + uzfwc) / (uztwm + uzfwm);
        perc  = perc * (1.0 + zp * np.power(defr,rexp));
        
        # NOTE...PERCOLATION OCCURS FROM UZFWC BEFORE PAV IS ADDED */
    
        if perc >= uzfwc:  # PERCOLATION RATE EXCEEDS UZFWC */
            perc = uzfwc;
        
        uzfwc = uzfwc - perc; # PERCOLATION RATE IS LESS THAN UZFWC.  */
    
        check = lztwc + lzfpc + lzfsc + perc - lztwm - lzfpm - lzfsm;
        if check > 0.:  # CHECK TO SEE IF PERCOLATION
            # EXCEEDS LOWER ZONE DEFICIENCY.
            perc = perc - check;
            uzfwc = uzfwc + check;
        
        #   SPERC IS THE TIME INTERVAL SUMMATION OF PERC
        sperc = sperc + perc;
        #        COMPUTE INTERFLOW AND KEEP TRACK OF TIME INTERVAL SUM.
        #        NOTE...PINC HAS NOT YET BEEN ADDED */
        delt = uzfwc * duz;
        sif =  sif + delt;
        uzfwc = uzfwc - delt;
        #        DESCRIBE PERCOLATED WATER INTO THE LOWER ZONES
        #          TENSION WATER MUST BE FILLED FIRST EXCEPT FOR THE
        #          PFREE AREA.  PERCT IS PERCOLATION TO TENSION WATER
        #          AND PERCF IS PERCOLATION GOING TO FREE WATER.  */
        perct = perc * (1.0 - pfree);
        if (perct + lztwc) <= lztwm:
            lztwc = lztwc + perct;
            percf = 0.0;
        else:
            percf = perct + lztwc - lztwm;
            lztwc = lztwm;
        
            #        DISTRIBUTE PERCOLATION IN EXCESS OF TENSION
            #        REQUIREMENTS AMONG THE FREE WATER STORAGES */
        percf = percf + (perc * pfree);
        if percf != 0:
    
            #           HPL IS THE RELATIVE SIZE OF THE PRIMARY STORAGE
            #           AS COMPARED WITH TOTAL LOWER ZONE FREE WATER STORAGE.  */
            hpl = lzfpm / (lzfpm + lzfsm);
    
            #           RATLP AND RATLS ARE CONTENT TO CAPACITY RATIOS, OR
            #           IN OTHER WORDS, THE RELATIVE FULLNESS OF EACH STORAGE */
            ratlp = lzfpc / lzfpm;
            ratls = lzfsc / lzfsm;
            #           FRACP IS THE FRACTION GOING TO PRIMARY.
            fracp = hpl * 2.0 * (1.0-ratlp) / (1.0-ratlp+1.0-ratls);
            if fracp > 1.0: fracp = 1.0
    
            #            PERCP AND PERCS ARE THE AMOUNT OF THE EXCESS
            #            PERCOLATION GOING TO PRIMARY AND SUPPLEMENTAL
            #            STORAGES, RESPECTIVELY
    
            percp = percf * fracp;
            percs = percf - percp;
            lzfsc = lzfsc + percs;
            if lzfsc > lzfsm:
                percs = percs + lzfsc + lzfsm;
                lzfsc = lzfsm;
            
            lzfpc = lzfpc + percf - percs;
    
            #           CHECK TO MAKE SURE LZFPC DOES NOT EXCEED LZFPM */
            if lzfpc > lzfpm:
                excess = lzfpc - lzfpm;
                lztwc = lztwc + excess;
                lzfpc = lzfpm;
            
        
        #                  DISTRIBUTE PINC BETWEEN UZFWC AND SURFACE RUNOFF
    
        if pinc != 0:
            if (pinc + uzfwc) <= uzfwm: # CHECK IF PINC EXCEEDS UZFWM
                uzfwc = uzfwc + pinc; # NO SUFACE RUNOFF
            else:
                sur = pinc + uzfwc - uzfwm;
                uzfwc = uzfwm;
                #               ADSUR IS THE AMOUNT OF SURFACE RUNOFF WHICH COMES
                #               FROM THAT PORTION OF ADIMP WHICH IS NOT
                #               CURRENTLY GENERATING DIRECT RUNOFF.  ADDRO/PINC
                #               IS THE FRACTION OF ADIMP CURRENTLY GENERATING
                #               DIRECT RUNOFF
                ssur = ssur + (sur * parea);
                adsur = sur * (1.0 - addro / pinc);
                ssur = ssur + adsur*adimp;
            
        
        adimc = adimc + pinc - addro - adsur;
        if adimc > (uztwm + lztwm):
            addro = addro + adimc - uztwm + lztwm;
            adimc = uztwm + lztwm;
          
    #    end
    #    END OF INCREMENTAL FOR LOOP
    
    #    COMPUTE SUMS AND ADJUST RUNOFF AMOUNTS BY THE AREA OVER
    #    WHICH THEY ARE GENERATED. */
    
    #    EUSED IS THE ET FROM PAREA WHICH IS 1.0-ADIMP-PCTIM */
    eused = e1 + e2 + e3;
    sif = sif * parea;        


    #    SEPARATE CHANNEL COMPONENT OF BASEFLOW FROM THE
    #    NON-CHANNEL COMPONENT */
    
    tbf = sbf * parea; # TBF IS TOTAL BASEFLOW */
    #    BFCC IS BASEFLOW, CHANNEL COMPONENT */
    bfcc = tbf * (1.0 / (1.0 + side));
    bfp = (spbf * parea) / (1.0 + side);
    bfs = bfcc - bfp;

    if bfs < 0: bfs = 0
    bfncc = tbf - bfcc; # BFNCC IS BASEFLOW, NON-CHANNEL COMPONENT */

    #    ADD TO MONTHLY SUMS. */
    sintft = sintft + sif;
    sgwfp = sgwfp + bfp;
    sgwfs = sgwfs + bfs;
    srecht = srecht + bfncc;
    srost = srost + ssur;
    srodt = srodt + sdro;

    #    COMPUTE TOTAL CHANNEL INFLOW FOR THE TIME INTERVAL.  */
    #tlci = roimp + sdro + ssur + sif + bfcc;
    tlci1 = roimp + sdro + ssur + sif;
    
    #    COMPUTE E4-ET FROM RIPARIAN VEGETATION.
    e4 = (edmnd - eused) * riva;
    
    #    SUBTRACT E4 FROM CHANNEL INFLOW */
    tlci1 = tlci1 - e4;
    
    if tlci1 < 0:
        e4 = e4 + tlci1;
        tlci1 = 0.;
    
    srot = srot + tlci1;
    
    # PERFORM SERIES OF LINEAR RESERVIOUR ROUTING
    UQ = tlci1;  # quickflow contribution         
    # Route quick flow component with 3 linear reservoirs in series
    xquick1 = (1-Kq)*xquick1 + (1-Kq)*UQ;
    outflow1 = (Kq/(1-Kq))*xquick1;
     
    xquick2 = (1-Kq)*xquick2 + (1-Kq)* outflow1;
    outflow2 = (Kq/(1-Kq))*xquick2;  
 
    xquick3 = (1-Kq)*xquick3 + (1-Kq)* outflow2;
    outflow3 = (Kq/(1-Kq))*xquick3;     

    #   CHANNEL INFLOW */
    tlci = outflow3 + bfcc;    

    #    COMPUTE TOTAL EVAPOTRANSPIRATION-TET */
    eused = eused * parea;
    tet = eused + e5 + e4;
    sett = sett + tet;
    se1 = se1 + (e1 * parea);
    se3 = se3 + (e3 * parea);
    se4 = se4 + e4;
    se5 = se5 + e5;

    #   CHECK THAT ADIMC >= UZTWC */
    if adimc < uztwc: adimc = uztwc;
    
    Stat = [uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc, xquick1, xquick2, xquick3]
    Stat = np.array(Stat).ravel()

    return tlci, Stat


def sacrun(par, states, Rain, PET, nens):


    flux = np.zeros([nens, 1])
    for time in range(4):
        for J in range(nens):
            pxv = Rain[time, J]  # precipitation in mm/6hr
            ep = PET[time, J]    # potential evapotranspiration in mm/6hr
            ##PERFORM SOIL MOISTURE ACCOUNTING OPERATIONS##
            par = np.vstack((par, [[0],[.3],[0]]))
            
            tlci, states[:, J] = fland1(ep, pxv, par, states[:,J])
            
            #SET TOTAL CHANNEL INFLOW EQUAL TO THE EXCESS AT THE END
            if tlci<0:tlci=0;
            flux[J] = flux[J]+np.real(tlci);        
            
            #states = np.where(states<0, 0, states)
            #states = np.where(np.isnan(states), 0, states)
    
            #flux = np.where(flux<0, 0, flux)
            #flux = np.where(np.isnan(flux), 0, flux)        
            
            
    return states, flux
        
        
        
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


