import numpy as np
import astropy.units as u
from astropy.constants import G
import coordinates

class lgkepler:
    '''
    This class defines the keplerian plane of the local group. 
    The 
    -------
    '''
    
    def __init__(self, semiMajorAxis, eccentricity, eccentricAnomaly, totalMass):
         # initializing the cosmology
        self.a    = semiMajorAxis * u.kpc    # in kpc
        self.e    = eccentricity     # dimensionless
        self.eta  = eccentricAnomaly # dimensionless
        self.Mtot = totalMass * u.Msun       # in Msun
        self.G    = G.to(u.kpc**3 / u.s**2 / u.Msun) # in kpc^3/s^2/Msun
    
    @property
    def separation(self):
        return self.a * ( 1 - self.e * np.cos(self.eta) )
    
    @property
    def time(self):
        A = self.a**3 / (self.G * self.Mtot)
        B = self.eta - ( self.e * np.sin(self.eta) )
        return A**(-1/2) * B
    
    @property
    def vrad_kepler(self):
        A = self.a / (self.G * self.Mtot)
        B = self.e * np.sin(self.eta)
        C = 1 - (self.e*np.cos(self.eta))
        return A**(-1/2) * B/C
    
    @property
    def vtan_kepler(self):
        A = self.a / (self.G * self.Mtot)
        B = ( 1 - self.e**2 )**(1/2)
        C = 1 - ( self.e * np.cos(self.eta) )
        return A**(-1/2) * B/C
    
    @property
    def trueAnomaly(self):
        A = ( 1+self.e )/(1-self.e )
        B = np.tan( self.eta / 2 )
        return 2 * np.arctan( np.sqrt(A) * B )
    
    @property
    def xy(self):
        return self.separation * np.cos( self.trueAnomaly ), self.separation * np.sin( self.trueAnomaly )

    @property
    def vxy(self):
        return self.vrad_kepler * np.cos( self.trueAnomaly ) - self.vtan_kepler * np.sin( self.trueAnomaly ),  self.vrad_kepler * np.sin( self.trueAnomaly ) + self.vtan_kepler * np.cos( self.trueAnomaly )
    
    