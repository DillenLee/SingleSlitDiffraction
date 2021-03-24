#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:06:56 2021

@author: 
   ___  _ ____             __          
  / _ \(_) / /__ ___      / /  ___ ___ 
 / // / / / / -_) _ \    / /__/ -_) -_)
/____/_/_/_/\__/_//_/   /____/\__/\__/ 

"""
#import necessary modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import curve_fit


#import the data
distance, voltage = np.loadtxt('data.csv', delimiter=',',unpack=True)
distance  = [x-15 for x in distance]
    
    

f = 500
λ = 670e-6
    
#Units are all in mm
def intensityFunc(x,a,I0):
    arg = (a*x)/(λ*f)
    return I0*((np.sinc(arg)**2))



#error analysis
#Take into account the error of the laser
def error(x,voltReading):
    #first calculate the wavelength error
    if x == 0:
        wavelengthError = 0
    else:
        wavelengthError = np.abs(((-3*x*np.cos(200*x)+(3/200)*np.sin(200*x))*np.sinc(200*x))/(900*x))     
    #then calculate multimeter error
    voltmeterError = (voltReading*0.05+1)
    print(voltmeterError)
    return wavelengthError+voltmeterError


#split into two bars for the assymetric error of the ambient light
upperErr=[error(distance[i],voltage[i]) for i in range(len(voltage))]
lowerErr=[val-1 for val in upperErr]
totalErr = [upperErr,lowerErr]

#curve fitting the data

guess = [0.06,250]
fit,cov = curve_fit(intensityFunc,distance,voltage,guess,sigma=upperErr)

datafit = []
largesample = np.linspace(-19,19,100)

for val in largesample:
    datafit.append(intensityFunc(val, *fit))
    
print(np.sqrt(cov[0][0]))
print(fit[0])






#%%
#-------
#import the anaylsed data
distanceAnalysed, greyAnalysed = np.loadtxt('Slit3.csv',delimiter=',',skiprows=1,unpack=True)
pixelTommRatio = 1/192
distanceAnalysed = [(x*pixelTommRatio)-10/3 for x in distanceAnalysed]
greyAnalysed = [(x*0.10) for x in greyAnalysed]

#plot the functions

plt.plot(distanceAnalysed,greyAnalysed)

plt.plot(largesample, datafit)
plt.errorbar(distance,voltage,yerr=totalErr,fmt='r|',capsize=4)
plt.grid()
plt.title('Graph of Voltage measured as a function of distance from maxima')
plt.xlabel('Distance from the centre of the central maxima (mm)')
plt.ylabel('Voltage (mV)')
plt.legend(('Modeled fit', 'Data Points'))
