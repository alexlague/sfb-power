###
# gSFB theory module from CAMB
# includes window function and integral constraint
# only does monopole without RSD for now
###

import numpy as np
import camb
from scipy.special import roots_legendre, spherical_jn
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
from mcfit import xi2P, P2xi

class gsfb_model:
    def __init__(self, ks, nofz, zmin, zmax, lmax, bg=1.0, zeff=None, camb_params=None, nquad=1024, nq=128, qmax=10.0):
        """
        """
        self.nofz = nofz
        self.zmin = zmin
        self.zmax = zmax
        #self.kmin = kmin
        #self.kmax = kmax
        self.lmax = lmax
        self.nquad = nquad
        self.bg = bg
        self.nq = nq
        self.qmax = qmax
        self.ks = ks
        self.qs = np.geomspace(1e-4, self.qmax, self.nq)
        self.ls = np.arange(self.lmax+1)# +1 to include l=0, +2 to include jl''(kx)
        
        if zeff is not None:
            self.zeff = zeff
        else:
            self.zeff = quad(lambda z: z*nofz(z), zmin, zmax)[0] 
            self.zeff /= quad(lambda z: nofz(z), zmin, zmax)[0]
            print("Estimated effective redshift: ", round(self.zeff, 3))
        
        if camb_params is not None:
            self.camb_params = camb_params
        else:
            self.camb_params = {'H0':67.5, 'ombh2':0.022, 'omch2':0.122, 'mnu':0.06, 'As':2e-9, 'ns':0.965}
            print("Using default CAMB params")
        
        return


    def compute_W_tensor(self):
        """
        """
        self._run_camb()
        self._get_quad_weights()
        integrand = lambda x: x**2*self.nofchi(x)#/nofchi_norm
        self.W = _matmul_W(self.ls, self.ks, self.qs, self.xn, self.wn, integrand)
        
        return
    
    def get_gsfb_model(self, compute_W=True):
        """
        """
        if compute_W:
            self.compute_W_tensor()

        plin_interp = self.pk_interp(self.qs)
        
        self.Cl = np.zeros((len(self.ls), len(self.ks)))
        for i in range(self.lmax+1):
            self.Cl[i, :] = simpson(self.W[i, :, i, :]**2 * self.qs**2*plin_interp, x=self.qs)
            self.Cl[i, :] /= simpson(self.W[i, :, i, :]**2 * self.qs**2, x=self.qs)
        
        pk_win = np.sum((2*self.ls+1)*self.Cl.T, axis=1) / np.sum(2*self.ls+1)
        
        self.gsfb_multipoles = {'g':{'g':{0:self.bg**2*pk_win}}}
        
        return
    
    def apply_integral_constraint(self):
        """
        """
        
        return

    def _run_camb(self):
        """
        """
        self.pars = camb.set_params(**self.camb_params)
        self.results = camb.get_results(self.pars)
        self.pars.set_matter_power(redshifts=[self.zeff], kmax=2*self.qmax) # no z evolution for now
        self.results.calc_power_spectra(self.pars)

        # Update to functions of comoving distance
        self.nofchi = lambda x: self.nofz(self.results.redshift_at_comoving_radial_distance(x))

        kh, z, pk = self.results.get_matter_power_spectrum(minkh=1e-4, maxkh=2*self.qmax, npoints=5000)
        self.pk_interp = CubicSpline(kh, pk[0])
        
        return

    def _get_quad_weights(self):
        """
        """
        # Linear weights for SFB
        x, w = roots_legendre(self.nquad) # xi and wi on -1, 1
        self.xmin = self.results.comoving_radial_distance(self.zmin)
        self.xmax = self.results.comoving_radial_distance(self.zmax)
        self.xn = 0.5*(self.xmax-self.xmin)*x + 0.5*(self.xmax+self.xmin)
        self.wn = 0.5*(self.xmax-self.xmin)*w

        # Log-weights for integral constraint
        #x, w = roots_legendre(self.nk) # ki and wi on -1, 1 match number of output k
        #self.klog = np.sqrt(self.kmin*self.kmax) * (self.kmax / self.kmin)**(x/2)
        #self.wlog = self.klog * np.log(self.kmax/self.kmin)/2 * w
        return

# Outside functions


def _sph_jn_table(ls, ks, xn):
    ls = np.asarray(ls); L = len(ls)
    assert np.all(np.diff(ls) == 1), "ls must be contiguous"
    kx = np.outer(ks, xn)
    safe = np.where(kx == 0, 1.0, kx)               # avoid /0, fixed at the end
    up = np.empty((L,) + kx.shape); dn = np.empty_like(up)

    # upward: j_{l+1} = (2l+1)/x j_l - j_{l-1}   (stable for l <= x)
    up[0] = spherical_jn(ls[0], kx); up[1] = spherical_jn(ls[1], kx)
    for i in range(1, L-1):
        up[i+1] = (2*ls[i]+1)/safe*up[i] - up[i-1]

    # downward: j_{l-1} = (2l+1)/x j_l - j_{l+1}   (stable for l > x)
    dn[-1] = spherical_jn(ls[-1], kx); dn[-2] = spherical_jn(ls[-2], kx)
    for i in range(L-2, 0, -1):
        dn[i-1] = (2*ls[i]+1)/safe*dn[i] - dn[i+1]

    J = np.where(ls[:, None, None] <= kx[None], up, dn)   # per-(l, kx) choice
    J[:, kx == 0] = (ls == 0)[:, None]                     # j_0(0)=1, j_l>0(0)=0
    return J

def _matmul_W(ls, ks, qs, xn, wn, integrand):
    Jk = _sph_jn_table(ls, ks, xn)
    Jq = _sph_jn_table(ls, qs, xn)
    F = wn*integrand(xn) # 1D
    L, K, X = Jk.shape          # Jk[l, k, x] = j_l(k x)
    M, Q, _ = Jq.shape          # Jq[m, q, x] = j_m(q x)

    out = ((Jk.reshape(L*K, X) * F) @ Jq.reshape(M*Q, X).T).reshape(L, K, M, Q)
    return out