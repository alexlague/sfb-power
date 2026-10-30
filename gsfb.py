r"""
Generalised spherical Fourier-Bessel (gSFB) decomposition of a field
binned into HEALPix shells.

    F^L_{lm}(k) = \int d^3x  j_L(k x) Y^*_{lm}(\hat x) F(x)

Each map is taken to be the field averaged over a radial bin [e_i, e_{i+1}], so

    1. angular:  d_{lm,i} = \int dOmega Y^*_{lm} F_i(\hat x)                 -> hp.map2alm per shell
    2. radial:   F^L_{lm}(k) = sum_i R[L,k,i] d_{lm,i}
                 R[L,k,i]   = \int_{e_i}^{e_{i+1}} dchi chi^2 j_L(k chi)        -> Gauss-Legendre per bin

R is returned so the same weights can be used in the model.

gsfb_cls then turns F^L_{lm}(k) into mask-decoupled angular spectra C_l^L(k) with pywiggle.
"""
import numpy as np
import healpy as hp
import pywiggle
from scipy.special import spherical_jn, gammaln


def gsfb_decompose(maps, chi_edges, k, lmax=None, Lmax=None, n_quad=None):
    """
    maps      : (n_shell, n_pix) HEALPix maps (RING), one per radial bin.
    chi_edges : (n_shell + 1,) comoving distances of the bin edges, increasing.
    k         : (n_k,) wavenumbers, in inverse units of chi.
    n_quad    : Gauss-Legendre nodes per bin (default: enough to resolve j_L across the widest bin).

    Returns dict with 'flm' of shape (n_k, Lmax+1, n_alm), labels 'k', 'L', 'l', 'm'
    (last axis in healpy alm ordering, m >= 0), and the radial weights 'R' of shape (Lmax+1, n_k, n_shell).
    """
    maps, edges, k = np.atleast_2d(maps), np.asarray(chi_edges, dtype=float), np.atleast_1d(k)
    if len(edges) != len(maps) + 1:
        raise ValueError(f"need n_shell + 1 = {len(maps) + 1} bin edges, got {len(edges)}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("chi_edges must be strictly increasing")
    if lmax is None:
        lmax = 3 * hp.npix2nside(maps.shape[1]) - 1
    if Lmax is None:
        Lmax = lmax
    if n_quad is None:
        n_quad = 8 + int(np.ceil(k.max() * np.diff(edges).max() / np.pi))

    alm = np.array([hp.map2alm(m, lmax=lmax) for m in maps])   # (shell, alm)

    # Gauss-Legendre nodes in every bin: x has shape (shell, node)
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    half, mid = 0.5 * np.diff(edges), 0.5 * (edges[1:] + edges[:-1])
    x = mid[:, None] + half[:, None] * xg
    w = half[:, None] * wg * x**2

    L = np.arange(Lmax + 1)
    jL = spherical_jn(L[:, None, None, None], k[None, :, None, None] * x)  # (L, k, shell, node)
    R = np.sum(jL * w, axis=-1)                                           # (L, k, shell)
    flm = np.einsum("Lki,ia->kLa", R, alm)

    ell, m = hp.Alm.getlm(lmax)
    return {"flm": flm, "k": k, "L": L, "l": ell, "m": m, "R": R}


def gsfb_decompose_photoz(maps, chi_centers, chi_sigmas, k, lmax=None, Lmax=None, n_quad=None, n_sigma=6.0):
    r"""
    As gsfb_decompose, but each shell has a Gaussian radial window (photo-z scatter):

        W_i(chi) = dchi_i * N(chi; chi_center_i, chi_sigma_i),   dchi_i = spacing of the centres

    so that sum_i W_i ~ 1 for evenly spaced shells (the analogue of top-hat bins), and

        R[L,k,i] = \int dchi chi^2 j_L(k chi) W_i(chi)       (Gauss-Legendre over +-n_sigma, chi >= 0)

    maps        : (n_shell, n_pix) HEALPix maps (RING).
    chi_centers : (n_shell,) window centres, increasing.
    chi_sigmas  : (n_shell,) or scalar, window widths (same units as chi).
    """
    maps, k = np.atleast_2d(maps), np.atleast_1d(k)
    c = np.asarray(chi_centers, dtype=float)
    s = np.broadcast_to(np.asarray(chi_sigmas, dtype=float), c.shape)
    if len(c) != len(maps):
        raise ValueError(f"need one centre per shell ({len(maps)}), got {len(c)}")
    if np.any(s <= 0):
        raise ValueError("chi_sigmas must be positive")
    if len(c) > 1 and np.any(np.diff(c) <= 0):
        raise ValueError("chi_centers must be strictly increasing")
    if lmax is None:
        lmax = 3 * hp.npix2nside(maps.shape[1]) - 1
    if Lmax is None:
        Lmax = lmax
    if n_quad is None:
        n_quad = 16 + int(np.ceil(k.max() * 2 * n_sigma * s.max() / np.pi))

    alm = np.array([hp.map2alm(m, lmax=lmax) for m in maps])   # (shell, alm)

    # integration range per shell, clipped at chi = 0
    lo, hi = np.maximum(c - n_sigma * s, 0.0), c + n_sigma * s
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
    x = mid[:, None] + half[:, None] * xg                                # (shell, node)
    dchi = np.gradient(c) if len(c) > 1 else np.ones(1)
    gauss = np.exp(-0.5 * ((x - c[:, None]) / s[:, None]) ** 2) / (np.sqrt(2 * np.pi) * s[:, None])
    w = half[:, None] * wg * x**2 * dchi[:, None] * gauss

    L = np.arange(Lmax + 1)
    jL = spherical_jn(L[:, None, None, None], k[None, :, None, None] * x)  # (L, k, shell, node)
    R = np.sum(jL * w, axis=-1)                                           # (L, k, shell)
    flm = np.einsum("Lki,ia->kLa", R, alm)

    ell, m = hp.Alm.getlm(lmax)
    return {"flm": flm, "k": k, "L": L, "l": ell, "m": m, "R": R}


def gsfb_cls(out1, out2=None, mask_alm=None):
    """
    Mask-decoupled, unbinned angular power spectra of the gSFB coefficients at each k and L:

        C_l^L(k) = < F1^L_{lm}(k) F2^L_{lm}(k)^* >,   l = 0..lmax   (decoupled with pywiggle)

    out1, out2 : outputs of gsfb_decompose / gsfb_decompose_photoz (out2=None -> auto-spectra).
                 They must share k, L and lmax.
    mask_alm   : alm of the angular mask applied to every shell, to at least 2*lmax
                 (None -> full sky). The mask must be the same for all shells.

    Returns dict with 'cl' of shape (n_k, Lmax+1, lmax+1), 'k', 'L', 'ell'.
    """
    out2 = out1 if out2 is None else out2
    f1, f2 = out1["flm"], out2["flm"]
    if f1.shape != f2.shape or not np.allclose(out1["k"], out2["k"]):
        raise ValueError("out1 and out2 must have the same k, L and lmax")
    n_k, n_L, _ = f1.shape
    lmax = int(out1["l"].max())

    if mask_alm is None:
        mask_alm = np.zeros(hp.Alm.getsize(2 * lmax), dtype=complex)
        mask_alm[0] = np.sqrt(4 * np.pi)
    if hp.Alm.getlmax(len(mask_alm)) < 2 * lmax:
        raise ValueError(f"mask_alm must extend to 2*lmax = {2 * lmax}")

    w = pywiggle.Wiggle(lmax, verbose=False)        # no bin_edges -> one bin per ell
    w.add_mask("mask", mask_alm)

    cl = np.array([[w.decoupled_cl(pywiggle.alm2cl(f1[ik, iL], f2[ik, iL]), "mask")["Cls"]
                    for iL in range(n_L)] for ik in range(n_k)])
    return {"cl": cl, "k": out1["k"], "L": out1["L"], "ell": np.arange(lmax + 1)}


# ----------------------------------------------------------------------------------------------
# Power spectrum multipoles from gSFB coefficients (arXiv:2404.04812, Eq. 6)
# ----------------------------------------------------------------------------------------------

def catalog_shell_maps(gal, rand, chi_edges, nside, mask=None, weights=None, subtract_randoms=True):
    """
    Bin a galaxy and a random catalogue into HEALPix shells, in the units gsfb_decompose expects:

        map_i(n) = [sum_{g in pixel} w_g - alpha N_r(n)] / (Omega_pix V_i),   V_i = (e_{i+1}^3 - e_i^3) / 3

    i.e. a (weighted) number density per unit volume, so that
    sum_i R[a,k,i] alm_i ~ int d^3x j_a Y^* (n_g^w - alpha n_r).

    gal, rand        : tuples (chi, ra, dec) in degrees.
    mask             : angular mask M(n) at this nside (None -> full sky), used for nbar and I22.
    weights          : per-galaxy weights w_g (None -> 1). For the line-of-sight momentum field
                       pass the radial velocities (times any other per-galaxy weight).
    subtract_randoms : subtract alpha x the random counts (density field). Set False for the
                       momentum field, whose expectation vanishes, so no random term is needed.

    alpha = sum(w_g) / N_r when subtracting randoms (weighted density), else N_g / N_r
    (velocity weights average to ~0, so the unweighted count sets nbar).

    Returns dict with 'maps' (n_shell, n_pix), 'shot' (n_shell,) the Poisson noise of each map,
    [sum_g w_g^2 + alpha^2 N_r] / V_i^2 (no random term without subtraction),
    'I22' = int nbar^2 M^2 d^3x, 'nbar' (n_shell,) and 'alpha'.
    """
    edges = np.asarray(chi_edges, dtype=float)
    npix, n_shell = hp.nside2npix(nside), len(edges) - 1
    omega_pix = 4 * np.pi / npix
    V = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    M = np.ones(npix) if mask is None else np.asarray(mask, dtype=float)

    def index(cat):
        chi, ra, dec = (np.asarray(c) for c in cat)
        ok = (chi >= edges[0]) & (chi < edges[-1])
        return ok, np.digitize(chi[ok], edges) - 1, hp.ang2pix(nside, ra[ok], dec[ok], lonlat=True)

    okg, ig, pg = index(gal)
    _, ir, pr = index(rand)
    wg = np.ones(len(ig)) if weights is None else np.asarray(weights, dtype=float)[okg]
    Nr = np.bincount(ir, minlength=n_shell)
    alpha = (wg.sum() if subtract_randoms else len(ig)) / len(ir)

    nbar = alpha * Nr / (V * omega_pix * M.sum())
    I22 = np.sum(nbar ** 2 * V) * omega_pix * np.sum(M ** 2)

    maps = np.empty((n_shell, npix))
    for i in range(n_shell):
        sel = ig == i
        counts = np.bincount(pg[sel], weights=wg[sel], minlength=npix)
        if subtract_randoms:
            counts -= alpha * np.bincount(pr[ir == i], minlength=npix)
        maps[i] = counts / (omega_pix * V[i])

    shot_w = np.bincount(ig, weights=wg ** 2, minlength=n_shell)
    if subtract_randoms:
        shot_w = shot_w + alpha ** 2 * Nr
    return {"maps": maps, "shot": shot_w / V ** 2, "I22": I22, "nbar": nbar, "alpha": alpha}


def cross_shot(gal, chi_edges, weights1=None, weights2=None, rand=None, alpha1=None, alpha2=None):
    """
    Poisson shot noise of the cross-spectrum between two shell-map sets built from the SAME
    galaxy catalogue with different weights (e.g. density, weights1=None, and momentum, weights2=v_r):

        shot_i = [sum_{g in shell i} w1_g w2_g + alpha1 alpha2 N_r,i] / V_i^2

    The random term is only included when both fields subtracted the same randoms: pass rand and
    both alphas (catalog_shell_maps()['alpha']). Leave rand=None if either field did not subtract
    randoms (as for the momentum field). Fields with no objects in common have no shot noise.
    """
    edges = np.asarray(chi_edges, dtype=float)
    n_shell = len(edges) - 1
    V = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    chi = np.asarray(gal[0])
    ok = (chi >= edges[0]) & (chi < edges[-1])
    ig = np.digitize(chi[ok], edges) - 1
    w1 = np.ones(ok.sum()) if weights1 is None else np.asarray(weights1, dtype=float)[ok]
    w2 = np.ones(ok.sum()) if weights2 is None else np.asarray(weights2, dtype=float)[ok]
    shot_w = np.bincount(ig, weights=w1 * w2, minlength=n_shell)
    if rand is not None:
        chir = np.asarray(rand[0])
        okr = (chir >= edges[0]) & (chir < edges[-1])
        shot_w = shot_w + alpha1 * alpha2 * np.bincount(np.digitize(chir[okr], edges) - 1, minlength=n_shell)
    return shot_w / V ** 2


def _wigner3j_000_sq(a, L, b):
    """(a L b; 0 0 0)^2, vectorised over integer arrays."""
    a, L, b = np.broadcast_arrays(*(np.asarray(x, dtype=int) for x in (a, L, b)))
    J = a + L + b
    ok = (J % 2 == 0) & (np.abs(a - b) <= L) & (L <= a + b)
    g = J // 2
    a_, L_, b_, J_, g_ = (np.where(ok, x, 0) for x in (a, L, b, J, g))
    lg = (gammaln(J_ - 2 * a_ + 1) + gammaln(J_ - 2 * L_ + 1) + gammaln(J_ - 2 * b_ + 1) - gammaln(J_ + 2)
          + 2 * (gammaln(g_ + 1) - gammaln(g_ - a_ + 1) - gammaln(g_ - L_ + 1) - gammaln(g_ - b_ + 1)))
    return np.where(ok, np.exp(lg), 0.0)


def gsfb_multipoles(out, I22, max_multipole=4, shot=None, nside=None, angle_average=False, out2=None,
                    subtract_shot=True):
    r"""
    Power spectrum multipoles P_L(k), L = 0..max_multipole, from the gSFB coefficients (Eq. 6):

        P_L(k) = (4pi)^2 (2L+1)/I22  sum_{a,b} i^{b-a} (2a+1) (a L b; 0 0 0)^2  D_b^{ab}(k)
        D_b^{ab}(k) = (2b+1) C_b^{ab,W}(k,k) = sum_m F1^a_{bm}(k) F2^{b*}_{bm}(k)

    a is the Bessel order (out['L'] axis), b the angular multipole. C^W is the window-convolved
    (pseudo) spectrum: the mask enters only through I22, so no mask decoupling is applied.
    Auto-spectrum when out2 is None (F2 = F1); cross-spectrum otherwise.

    out           : field 1, output of gsfb_decompose / gsfb_decompose_photoz, computed with
                    Lmax >= lmax + max_multipole (the Bessel order reaches b + L). Field 1 carries
                    the Bessel order a, i.e. it plays the role of F_L in a Yamamoto estimator, so
                    P^{12}_L and P^{21}_L differ (odd L change sign in the plane-parallel limit).
    I22           : normalisation. Auto: int nbar^2 W^2 d^3x. Cross: the overlap int nbar1 nbar2 W1 W2 d^3x
                    (same galaxies -> the same I22). Do not use sqrt(I22_1 * I22_2): it is >= the overlap
                    and biases the cross-spectrum low when the samples or footprints differ.
    shot          : (n_shell,) Poisson noise per shell map. Auto: catalog_shell_maps()['shot'].
                    Cross: cross_shot(...) when both fields share objects; None when they don't.
                    Shot spectrum D_shot = (2b+1)/(4pi) sum_i shot_i R1[a,k,i] R2[b,k,i], returned as 'P_shot'.
    nside         : if given, deconvolve the HEALPix pixel window (after removing shot noise,
                    which is not smoothed by pixelisation).
    angle_average : divide by 4pi (standard Yamamoto convention, int dOmega_k / 4pi).
    out2          : field 2 (same k, lmax and Bessel orders as out), or None for the auto-spectrum.
    subtract_shot : if True (default) 'P' is shot-noise subtracted; if False 'P' keeps the shot noise,
                    i.e. P = P_signal + P_shot (the shot part is not pixel-window deconvolved).

    Returns dict with 'P' (max_multipole+1, n_k) complex: even L real, odd L carry the factor i
    (e.g. the galaxy-momentum cross-spectrum lives mainly in the odd multipoles), 'k', 'multipoles',
    and, when shot is given, 'P_shot': the expected Poisson shot-noise multipoles in the same convention
    (add it to the model if you keep the shot noise in the data).
    """
    out2 = out if out2 is None else out2
    F1, R1, F2, R2 = out["flm"], out["R"], out2["flm"], out2["R"]
    if F1.shape != F2.shape or not np.allclose(out["k"], out2["k"]):
        raise ValueError("out and out2 must share k, lmax and Bessel orders")
    ell, m = out["l"], out["m"]
    lmax, A_max = int(ell.max()), F1.shape[1] - 1
    if A_max < lmax + max_multipole:
        raise ValueError(f"need Bessel orders up to lmax + max_multipole = {lmax + max_multipole}; "
                         f"rerun the decomposition with Lmax={lmax + max_multipole}")
    n_k = F1.shape[0]
    b = np.arange(lmax + 1)
    wm = np.where(m == 0, 1.0, 2.0)                   # healpy stores m >= 0 only
    idx = np.arange(len(ell))
    pw2 = hp.pixwin(nside, lmax=lmax) ** 2 if nside is not None else np.ones(lmax + 1)

    P = np.zeros((max_multipole + 1, n_k), dtype=complex)
    P_shot = np.zeros_like(P)
    for d in range(-max_multipole, max_multipole + 1):          # d = a - b
        ok = ell + d >= 0
        # sum over m = -b..b of F1 F2^*, using F_{b,-m} = (-1)^m F_{bm}^* for real fields
        prod = (F1[:, ell[ok] + d, idx[ok]] * np.conj(F2[:, ell[ok], idx[ok]])).real * wm[ok]
        D = np.array([np.bincount(ell[ok], weights=p, minlength=lmax + 1) for p in prod])  # (k, b)
        if shot is not None:
            bv = b[b + d >= 0]
            Dshot = np.zeros((n_k, lmax + 1))
            Dshot[:, bv] = (2 * bv + 1) / (4 * np.pi) * np.einsum("bki,bki,i->kb", R1[bv + d], R2[bv], shot)
            D = D - Dshot
        D = D / pw2
        for Lp in range(abs(d), max_multipole + 1, 2):           # parity: a + L + b even
            a = b + d
            coef = np.where(a >= 0, (2 * a + 1) * _wigner3j_000_sq(np.maximum(a, 0), Lp, b), 0.0)
            P[Lp] += (1j) ** (-d) * (D @ coef)
            if shot is not None:
                P_shot[Lp] += (1j) ** (-d) * (Dshot @ coef)

    norm = (4 * np.pi) ** 2 * (2 * np.arange(max_multipole + 1) + 1)[:, None] / I22
    if angle_average:
        norm = norm / (4 * np.pi)
    P, P_shot = P * norm, P_shot * norm
    if shot is not None and not subtract_shot:
        P = P + P_shot
    res = {"P": P, "k": out["k"], "multipoles": np.arange(max_multipole + 1)}
    if shot is not None:
        res["P_shot"] = P_shot
    return res
