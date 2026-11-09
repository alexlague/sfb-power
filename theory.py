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
    def __init__(self, ks, nofz, zmin, zmax, lmax, bg=1.0, zeff=None, camb_params=None, nquad=1024, n_dq=4, qmax=None, max_sep=0, odd_sep=False):
        """
        qmax    : largest q of the P(q) integral. None -> ks.max() + 50/(xmax - xmin), set once the distances are known
        max_sep : largest |l1 - l2| stored in the W dictionary (0 -> diagonal only, enough for the monopole)
        odd_sep : also store odd l1 - l2 (needed only for odd multipoles, e.g. galaxy-momentum cross-spectra)
        """
        self.max_sep = max_sep
        self.odd_sep = odd_sep
        self.nofz = nofz
        self.zmin = zmin
        self.zmax = zmax
        #self.kmin = kmin
        #self.kmax = kmax
        self.lmax = lmax
        self.nquad = nquad
        self.bg = bg
        self.n_dq = n_dq # replace nq: number of points per oscillation
        self.qmax = qmax
        self.ks = ks
        #self.qs = np.geomspace(1e-4, self.qmax, self.nq)
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
        self.qs = _make_q_grid(self.qmax, self.xmax, self.n_dq)
        integrand = lambda x: x**2*self.nofchi(x)#/nofchi_norm
        wf = self.wn*integrand(self.xn)
        # len(self.ls) = lmax+1, so l2 covers 0..lmax like self.ls, and l1 = l2 + d reaches lmax + max_sep
        self.W = _compute_W_dict(len(self.ls), self.max_sep, self.ks, self.qs, self.xn, wf, self.xmax,
                                 odd=self.odd_sep)
        
        return
    
    def get_gsfb_model(self, compute_W=True):
        """
        """
        if compute_W:
            self.compute_W_tensor()

        plin_interp = self.pk_interp(self.qs)
        
        self.Cl = np.zeros((len(self.ls), len(self.ks)))
        #for i in range(self.lmax+1):
        #    self.Cl[i, :] = simpson(self.W[i, :, i, :]**2 * self.qs**2*plin_interp, x=self.qs)
        #    self.Cl[i, :] /= simpson(self.W[i, :, i, :]**2 * self.qs**2, x=self.qs)

        for i in range(self.lmax+1):
            self.Cl[i, :] = 2/np.pi*simpson(self.W[(i, i)]**2 * self.qs**2*plin_interp, x=self.qs)

        self.I22 = 4*np.pi*np.sum(self.wn*self.xn**2*self.nofchi(self.xn)**2)
        pk_win = 4*np.pi/self.I22*np.sum((2*self.ls+1)*self.Cl.T, axis=1)

        #pk_win = np.sum((2*self.ls+1)*self.Cl.T, axis=1) / np.sum(2*self.ls+1)
        
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

        # Distances in Mpc/h, needed before the default qmax can be set
        self.h = self.pars.h
        self.xmin = self.results.comoving_radial_distance(self.zmin)*self.h
        self.xmax = self.results.comoving_radial_distance(self.zmax)*self.h
        self.nofchi = lambda x: self.nofz(self.results.redshift_at_comoving_radial_distance(x/self.h))
        if self.qmax is None:
            self.qmax = self.ks.max() + 50/(self.xmax - self.xmin)

        self.pars.set_matter_power(redshifts=[self.zeff], kmax=2*self.qmax) # no z evolution for now
        self.results.calc_power_spectra(self.pars)

        kh, z, pk = self.results.get_matter_power_spectrum(minkh=1e-4, maxkh=2*self.qmax, npoints=5000)
        self.pk_interp = CubicSpline(kh, pk[0])
        
        return

    def _get_quad_weights(self):
        """
        """
        # Linear weights for SFB
        x, w = roots_legendre(self.nquad) # xi and wi on -1, 1
        # xmin and xmax (Mpc/h) are set in _run_camb
        self.xn = 0.5*(self.xmax-self.xmin)*x + 0.5*(self.xmax+self.xmin)
        self.wn = 0.5*(self.xmax-self.xmin)*w

        # Log-weights for integral constraint
        #x, w = roots_legendre(self.nk) # ki and wi on -1, 1 match number of output k
        #self.klog = np.sqrt(self.kmin*self.kmax) * (self.kmax / self.kmin)**(x/2)
        #self.wlog = self.klog * np.log(self.kmax/self.kmin)/2 * w
        return

# Outside functions


def _sph_jn_table(ls, ks, xn):
    """
    j_l(k x), shape (l, k, x), for ls = 0..lmax. Below l ~ kx use the upward recurrence. Above it,
    multiply by the ratio r_l = j_l/j_{l-1} from a downward continued fraction, which needs no seed
    value that can underflow.
    """
    ls = np.asarray(ls); lmax = int(ls[-1])
    assert ls[0] == 0 and lmax >= 1 and np.all(np.diff(ls) == 1), "ls must be 0..lmax with lmax >= 1"
    kx = np.outer(ks, xn)
    safe = np.where(kx == 0, 1.0, kx)               # avoid /0, fixed at the end
    J = np.empty((len(ls),) + kx.shape)

    r = np.zeros_like(kx)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for l in range(lmax + int(np.sqrt(40*(lmax+1))) + 10, 1, -1):
            r = 1/((2*l+1)/safe - r)                # r_l = j_l/j_{l-1}
            if l <= lmax:
                J[l] = r                            # overwritten by j_l below
        J[0] = spherical_jn(0, kx); J[1] = spherical_jn(1, kx)
        for l in range(1, lmax):
            J[l+1] = np.where(l <= kx, (2*l+1)/safe*J[l] - J[l-1], J[l+1]*J[l])
    J[:, kx == 0] = (ls == 0)[:, None]              # j_0(0)=1, j_l>0(0)=0
    return J

def _turning_mask(l, kq, xmax):
    """True where j_l(kq x) is not negligible somewhere in x <= xmax: kq*xmax > l - 8 l^(1/3)."""
    l = np.asarray(l, dtype=float)
    return kq[None, :]*xmax >= (l - 8*np.cbrt(l))[:, None]

def _compute_W_dict(lmax, L, k_array, q_array, x, wf, xmax, odd=False, chunk=128):
    """
    W[(l1, l2)][k, q] = sum_x wf(x) j_l1(k x) j_l2(q x),  for 0 <= l2 < lmax and l1 = l2 + d,
    with |d| <= L and l1 >= 0. Only even d are stored unless odd=True.

    l1 is the Bessel order on the k side and can reach lmax - 1 + L, so every band is complete
    up to l2 = lmax - 1. l2 is the angular order on the q side. Entries where j_l1(k x) or
    j_l2(q x) is negligible for all x <= xmax are not computed and stay zero. wf holds the
    quadrature weights times the integrand.

    The q-side Bessel table is built in chunks of q so its size stays (lmax, chunk, len(x)).
    """
    seps = [d for d in range(-L, L+1) if odd or d % 2 == 0]
    keys = [(l2 + d, l2) for l2 in range(lmax) for d in seps if l2 + d >= 0]

    lk = np.arange(max(lmax + L, 2))                   # k-side orders, up to lmax - 1 + L
    lq = np.arange(max(lmax, 2))
    mask_k = _turning_mask(lk, k_array, xmax)          # (l1, k)
    mask_q = _turning_mask(lq, q_array, xmax)          # (l2, q)
    jk = _sph_jn_table(lk, k_array, x)*wf              # (l1, k, x), weights folded in once

    W_dict = {key: np.zeros((len(k_array), len(q_array))) for key in keys}

    for i in range(0, len(q_array), chunk):
        e = min(i + chunk, len(q_array))
        need = np.nonzero(mask_q[:, i:e].any(axis=1))[0]
        if len(need) == 0:
            continue
        jq = _sph_jn_table(np.arange(max(need[-1], 1) + 1), q_array[i:e], x)   # (l2, chunk, x)
        iq = np.arange(i, e)
        for (l1, l2), W in W_dict.items():
            if l2 > need[-1]:
                continue
            mk, mq = mask_k[l1], mask_q[l2, i:e]
            if mk.any() and mq.any():
                W[np.ix_(mk, iq[mq])] = jk[l1][mk] @ jq[l2][mq].T

    return W_dict

def _make_q_grid(qmax, xmax, n_dq=4, qmin=1e-4, dlnq=0.02):
    dq = np.pi/(n_dq*xmax)
    qsw = min(dq/dlnq, qmax)
    qlog = np.exp(np.arange(np.log(qmin), np.log(qsw), dlnq))
    return np.append(qlog, np.arange(qsw, qmax+dq, dq))
