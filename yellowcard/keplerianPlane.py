import numpy as np
import astropy.units as u
from astropy.constants import G


class LGKepler:
    '''
    This class defines the keplerian plane of the local group.
    -------
    '''

    def __init__(self, semiMajorAxis, eccentricity, eccentricAnomaly, totalMass):
        self.a    = semiMajorAxis    # semimajor axis of fictious keplerian orbit
        self.e    = eccentricity     # eccentricity of fictious keplerian orbit
        self.eta  = eccentricAnomaly # eccentric anomaly of fictious keplerian orbit
        self.Mtot = totalMass        # total LG mass
        self.G    = G                # gravitational constant

        # try:
        #     self.a.unit
        # except AttributeError:
        #     print("Warning: Assuming units of semimajor axis are kpc")
        #     self.a *= u.kpc

        # try:
        #     self.Mtot.unit
        # except AttributeError:
        #     print("Warning: Assuming units of total mass are Msun")
        #     self.Mtot *= u.Msun

    @property
    def separation(self):
        return self.a * ( 1 - self.e * np.cos(self.eta) )

    @property
    def time(self):
        A = self.a / self.vscale
        if hasattr(self.eta, 'unit'):
            eta = self.eta.to_value(u.rad)
        else:
            eta = self.eta
        B = eta - ( self.e * np.sin(self.eta) )
        return A * B
    
    @property
    def vscale(self):
        A = (self.G * self.Mtot) / self.a 
        return np.sqrt(A)

    @property
    def vrad_kepler(self):
        A = self.vscale
        B = self.e * np.sin(self.eta)
        C = 1 - (self.e*np.cos(self.eta))
        return A * B/C

    @property
    def vtan_kepler(self):
        A = self.vscale
        B = ( 1 - self.e**2 )**(1/2)
        C = 1 - ( self.e * np.cos(self.eta) )
        return A * B/C

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

# attaching alternate def for xdot and ydot:
#     @property
#     def vxy(self):
#         A = np.sqrt(self.G*self.Mtot/self.a)
#         B = np.sin(self.trueAnomaly) / np.sqrt(1-self.e**2)
#         C = (self.e + np.cos(self.trueAnomaly)) / np.sqrt(1-self.e**2)
#         return -A * B, A*C

