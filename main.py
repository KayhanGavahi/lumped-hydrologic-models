# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:12:44 2022

@author: kgavahi
"""


from models.hymod import *
from models.sacrun import *
import time


xx1 = 29.5680077836428
xx2 = 5.52012021719543
xx3 = 5.25240932417414
xx4 = 4.84910004569187
xx5 = 5.10311727968659
Pars = [0.698934164644799, 0.581765444311757, 350.125542405752, 0.0766606739123965, 0.778234471633011]    
I = 0.0079602388712033
Pet = 4.49389442835338


s = time.time()
for i in range(10000):
    [xx1,xx2,xx3,xx4,xx5,qt] = hymod(xx1, xx2, xx3, xx4, xx5, Pars, I, Pet)
    
print(time.time()-s)    
    
nens = 1
par = [   58.9543478064958,
          26.7235837745124,
         0.334288585853897,
         0.198590504517122,
         0.226937519678351,
          87.4610381957428,
          4.78584668030303,
          91.4631926708218,
          923.682042651196,
          41.0908386516808,
         0.125489542053775,
         0.354555496319456,
         0.471339961640229,
        0.0177132241044721]
par = np.array(par).reshape(len(par), 1)
states = [67.6985951415985,
          11.7717490629483,
          177.246401081489,
         0.833558044224772,
           8.7316731399883,
          182.820054006262,
         0.458098942664452,
         0.513378516938732,
         0.485235391514304]
states = np.array(states).reshape(len(states), 1)
Rain = [0, 0, 0, 0]
Rain = np.array(Rain).reshape(len(Rain), 1)
PET = [0.291222609561752, 0.291222609561752, 0.291222609561752, 0.291222609561752]
PET = np.array(PET).reshape(len(PET), 1)

s = time.time()

states, flux = sacrun(par, states, Rain, PET, nens)
print(time.time()-s)  