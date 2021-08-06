# -*- coding: utf-8 -*-
"""SCDT.py

Class for computing the Signed Cumulative Distribution Transform (SCDT).

Authors: 
Sumati Thareja
D.M. Rocio
Ivan Medri
Akram Aldroubi
Gustavo Rohde

Based on paper:
ArXiv paper link
"""

import numpy as np
from numpy import interp
import math
#import matplotlib.pyplot as plt
#from scipy.linalg import lstsq
#from scipy import signal

"""## **Class : SCDT**"""

class SCDT:
    """
    Signed Cumulative Distribution Transform (SCDT)
    
    Parameters
    ----------

    reference: 1D array representing the reference density
    x0: domain of reference
    reference_CDF: reference CDF
    xtilde: Domain of the reference's inverse CDF 
    reference_CDT_inverse: inverse CDF of reference

    """

    def __init__(self, reference, x0=None):
        """
        the reference (or the reference density) is normalized, its CDF its CDF's inverse are calculated
        """
        assert not((1.0*reference<0).sum())
        reference = reference/reference.sum() #reference's normalization
        self.reference = reference
        self.dim = len(reference)
        self.reference_CDF = np.cumsum(reference) #reference's CDF
        self.x = np.linspace(0,1,self.dim) #Defining the domain of input signal
        self.xtilde = np.linspace(0,1,self.dim) #Domain of the reference signal
        if x0 is not None:
            self.xtilde = x0
        self.reference_CDF_inverse = interp(self.xtilde,self.reference_CDF,self.x) #Inverse of the references's CDF
   
   
    def gen_inverse(self,f,dom_f,dom_gf):
        """
        gen_inverse calculates the generalized inverse of the function f
        input:
            f: A one dimensional function represented by a vector
            dom_f: The domain of the function f
            dom_gf: The domain of the generalizd inverse of f
        output:
            The generalized inverse of f
        """
        infi = 0
        n = len(dom_f)
        #i=0 
        j=0
        gf=[0]*n
        k=0
        while j < n:
            #print("dom_gf = ",dom_gf[j])
            i=0
            y=[]
            while i < n:
             #   print(i,"th iteration")
                if f[i] > dom_gf[j] :
                    #print("f = ",f(dom_f[i])
                    y.append(dom_f[i])
                else :
                    y.append(math.inf)
               # print(y)
                i=i+1
            gf[j] = (min(y))
            #print("gf = ",gf[j])
            j=j+1
        return np.array(gf)
        
    def transform(self,I):
        """
        transform calculates the transport map that morphs the one-dimensional distribution I into the reference.
        input:
            I: A one dimensional distributions of size self.dim
        output:
            The CDT transformation of I
        """
        #assert self.dim==len(I)
        #assert not((1.0*I<=0).sum())
        #Force I to be a positive probability distribution
        #eps=1e-5 #This small dc level is needed for a numerically unique solution of the transport map
        #I=I+eps
        I=I/I.sum()
        #Calculate its CDF
        I_CDF=np.cumsum(I)
        #I_CDF_inverse = self.gen_inverse(I_CDF, self.x,self.xtilde) #Inverse of the CDF of I
        Ihat = self.gen_inverse(I_CDF,self.x,self.reference_CDF) #Composition I_CDF_inverse(reference_CDF(x))
        return Ihat
    
    def itransform(self,Ihat):
        """
        itransform calculates the inverse of the CDT. It receives a transport displacement
        and the reference, and finds the one dimensional distribution I from it.
        input:
            Transport displacement map
            The reference used for calculating the CDT
        output:
            I: The original distribution
        """
        I = interp(self.x, Ihat, self.reference_CDF)
        return I
    
    def stransform(self,I,x=None):
        """
        stransform calculates the transport transform (CDT) of a signal I for signals that may change sign
        input:
            The original density I
            x -> domain of the density I
        output:
            The 4 components of the transform of signed signals: the CDT of the positive and the negative 
            part of I, and the total masses of the positive and the negative part of I            
        """
        if x is not None:
            self.x = x
        eps = np.finfo(float).eps
        #Calculate the positive part of I
        Ipos = np.array([(abs(s)+s)/2 for s in I]) 
        #Calculate the negative part of I
        Ineg = np.array([(abs(s)-s)/2 for s in I]) 
        Iposhat = self.transform(Ipos) 
        Ineghat = self.transform(Ineg)
        return Iposhat, Ineghat, Ipos.sum(), Ineg.sum()
    
    def istransform(self,Ipos,Ineg,Masspos,Massneg):
        """
        istrasform calculates the inverse of the stransform.
        input:
            The 4 components of the transport transform for signed signals
        output:
            The original signal 
        """
        return self.itransform(Ipos)*Masspos-self.itransform(Ineg)*Massneg

