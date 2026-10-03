"""
OU networks and Hankel-based reachability / causality inference.

Companion code to the Methods section. Paper <-> code notation:

    l   odd number of lagged covariances      N = (l+1)/2 block rows,
                                              i.e. l = 2N - 1
    H[a,b] = Lambda(a+b+1), a,b = 0..N-1      hankel()
    W = <ff>^-1/2 <fp> <pp>^-1/2              whitened_hankel()
    gamma_1 >= gamma_2 >= ... (canonical
        correlations of past and future)      *_canonical_correlations()
    T_e = T - l                               effective sample size
    P(rho = r | Y)   BIC rank posterior       empirical_rank_posterior()
    log-odds of causality (Eq. main:proba)    empirical_kalman_causality_probability()

Every quantity comes in a THEORETICAL and an EMPIRICAL version:

    theoretical_*(sys, sensors, h, N)   from A and Sigma  (noise-free)
    empirical_*(X, cols, N)             from a finite record

Argument convention: theoretical_* take NODE LABELS (whatever the graph
uses), empirical_* take COLUMN INDICES into X, obtained as
[sys.index[v] for v in sensors].

Both Hankel forms start at lag 1, as in the text: the raw Hankel has
blocks Lambda(a+b+1) and the whitened form's cross block is
Cov(future, past) with the same lags; the lag-0 matrix enters only the
Toeplitz whitening factors.

Requires: numpy, scipy, networkx


ESTIMATION CONVENTIONS  (changed 2026-09; see CHANGES at the bottom)
--------------------------------------------------------------------
1. CONSTANT AVERAGING WINDOW.  Every empirical second moment is an
   average of exactly T_e = T - l products, matching the text's
   "<.> is the average over k = 1, ..., T - l".  Two estimators are
   provided and both respect it:

     empirical_lags()             one Lambda-hat per lag, each averaged
                                  over the same T_e window anchored at
                                  the same base index.  Feeds the RAW
                                  Hankel, which therefore keeps its
                                  exact block-Hankel/Toeplitz structure.

     empirical_moment_matrices()  the text's Eq. (16) taken literally:
                                  stack the past and future vectors for
                                  k = 1..T_e and form <ff^T>, <fp^T>,
                                  <pp^T> as sample second moments of
                                  those stacks.

   The two differ only in which T_e-length slice of the record each
   individual block is averaged over; both use T_e terms everywhere.

2. NO CLIPPING ANYWHERE.  There is no eigenvalue floor in the whitening
   and no clipping of the canonical correlations.  This is possible
   because <ff^T> and <pp^T> from empirical_moment_matrices() are Gram
   matrices of the stacked data, hence positive semi-definite exactly,
   and positive definite whenever T_e >= N*m and the record is in
   general position -- which also makes gamma in [0, 1] automatic by
   Cauchy-Schwarz.  When a matrix is NOT positive definite the code now
   raises with a diagnostic instead of silently regularising: that
   condition means the record is too short (or too collinear) for the
   chosen block depth, and it should be seen, not patched.
"""

import warnings

import numpy as np
import networkx as nx
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.signal import lfilter
from scipy.special import expit, logsumexp


# ======================================================================
# 1. OU network  (theory: Eqs. OU, partition, reduced, lyapunov,
#    covdynamics, lambda)
# ======================================================================
class OUNetwork:
    """
    dX = A X dt + D dW,  D = diag(sigma_1, ..., sigma_n), on a DiGraph.
    Edge convention: u -> v in G means u drives v, i.e. A[v, u] != 0.
    """

    def __init__(self, G, weight_range=(1.0, 2.0), sigma_range=(0.01, 0.05),
                 stabilize="shift", spectral_margin=0.3,
                 self_loop_margin=0.5, seed=None):
        """
        stabilize:
          'shift'     - free couplings, then a uniform diagonal shift
                        placing the spectral abscissa at -spectral_margin.
                        Coupling-to-damping ratios stay near 1, so distant
                        nodes remain visible (mild per-hop attenuation).
                        Slowest time constant is 1/spectral_margin.
          'dominance' - diagonal dominance (Gershgorin). Always stable but
                        every coupling is sub-dominant, giving strong
                        per-hop attenuation.
        """
        if not G.is_directed():
            raise TypeError(
                "G must be directed (nx.DiGraph). An undirected Graph "
                "yields each edge once as (u, v) with u < v, so A comes "
                "out triangular, while nx.ancestors() on the same object "
                "returns the whole component -- output_reachable_set() "
                "would then disagree with the system actually built. Use "
                "nx.erdos_renyi_graph(n, p, directed=True) or "
                "G.to_directed().")
        self.G = G
        self.nodes = list(G.nodes())
        self.index = {v: i for i, v in enumerate(self.nodes)}
        self.n = len(self.nodes)
        rng = np.random.default_rng(seed)

        A = np.zeros((self.n, self.n))
        for u, v in G.edges():
            if u == v:
                continue
            A[self.index[v], self.index[u]] = \
                rng.uniform(*weight_range) * rng.choice([-1.0, 1.0])

        if stabilize == "shift":
            for i in range(self.n):
                A[i, i] = -rng.uniform(0.5, 1.5)
            A -= (np.max(np.real(np.linalg.eigvals(A))) + spectral_margin) \
                * np.eye(self.n)
        elif stabilize == "dominance":
            for i in range(self.n):
                off = np.sum(np.abs(A[i])) - np.abs(A[i, i])
                A[i, i] = -(off + rng.uniform(self_loop_margin,
                                              2 * self_loop_margin))
        else:
            raise ValueError("stabilize must be 'shift' or 'dominance'")
        self.A = A

        self.sigma = rng.uniform(*sigma_range, size=self.n)
        self.D = np.diag(self.sigma)
        self.Sigma = solve_continuous_lyapunov(self.A, -self.D @ self.D)
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        self._rng = rng

    # ---------------- analytics ----------------
    def lagged_covariance(self, tau):
        """Gamma(tau) = Cov(X(t+tau), X(t)) = expm(A tau) Sigma, tau >= 0."""
        return expm(self.A * tau) @ self.Sigma

    def analytical_lags(self, sensors, h, n_lags):
        """Noise-free [Lambda(0), ..., Lambda(n_lags-1)] for the sensors."""
        cols = [self.index[s] for s in sensors]
        Ab = expm(self.A * h)
        out, M = [], np.eye(self.n)
        for _ in range(n_lags):
            out.append((M @ self.Sigma)[np.ix_(cols, cols)])
            M = Ab @ M
        return out

    def output_reachable_set(self, sensors):
        """Nodes with a directed path to some sensor, plus the sensors
        themselves (paths of length zero are allowed, so S is contained
        in P_S). Its size n_p is the generic Hankel rank (Theorem)."""
        P = set(sensors)
        for s in sensors:
            P |= nx.ancestors(self.G, s)
        return P

    # ---------------- simulation ----------------
    def simulate(self, T, h, seed=None):
        """
        Exact stationary sample at period h (no discretization error),
        vectorized via the eigendecomposition of expm(A h).
        Returns (T, n); column order follows self.nodes.
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        Ab = expm(self.A * h)
        Q = self.Sigma - Ab @ self.Sigma @ Ab.T
        Lq = _psd_factor(0.5 * (Q + Q.T), "innovation covariance Q")
        L0 = _psd_factor(self.Sigma, "stationary covariance Sigma")

        lam, V = np.linalg.eig(Ab)
        W = rng.standard_normal((T, self.n)) @ Lq.T
        W[0] = L0 @ rng.standard_normal(self.n)
        U = W @ np.linalg.inv(V).T
        Z = np.empty_like(U, dtype=complex)
        for i in range(self.n):              # z_k = lam_i z_{k-1} + u_k
            Z[:, i] = lfilter([1.0], [1.0, -lam[i]], U[:, i])
        return np.real(Z @ V.T)


def _psd_factor(S, name="matrix"):
    """
    A factor L with L L^T = S, for a matrix that is PSD in exact
    arithmetic.

    Cholesky first. If it fails, eigh: eigenvalues that are negative by
    more than round-off (|lam| > n * eps * lam_max) mean S is genuinely
    indefinite and raise; eigenvalues negative only at round-off level
    are set to exactly zero, which is the exact PSD projection of a
    matrix that is PSD up to floating point, and a warning is issued so
    the substitution is never silent.

    This is a SIMULATION helper (drawing the stationary state and the
    innovations); it is not on the estimation path.
    """
    S = 0.5 * (S + S.T)
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        pass
    w, V = np.linalg.eigh(S)
    w_max = w.max()
    tol = S.shape[0] * np.finfo(float).eps * max(w_max, 0.0)
    if w.min() < -tol:
        raise np.linalg.LinAlgError(
            f"{name} is indefinite (min eigenvalue {w.min():.3e}, "
            f"max {w_max:.3e}, round-off tolerance {tol:.3e}); it cannot "
            f"be factored as L L^T. This is a modelling problem, not a "
            f"numerical one.")
    warnings.warn(
        f"{name}: Cholesky failed; using the eigendecomposition with "
        f"{int((w < 0).sum())} eigenvalue(s) negative at round-off level "
        f"(min {w.min():.3e}) set to zero.", RuntimeWarning, stacklevel=2)
    return V @ np.diag(np.sqrt(np.where(w < 0.0, 0.0, w)))


# ======================================================================
# 2. Lag matrices  (Eq. lambda / Eq. samplecov)
# ======================================================================
def theoretical_lags(sys, sensors, h, n_lags):
    """Noise-free lagged covariances (thin wrapper, NODE LABELS)."""
    return sys.analytical_lags(sensors, h, n_lags)


def empirical_lags(X, cols, n_lags):
    """
    Lagged covariances on a CONSTANT window, for the given COLUMNS:

        Lambda_hat(k) = (1/T_e) sum_{s=0}^{T_e-1} y_{s+k} y_s^T,
        k = 0, ..., n_lags-1,        T_e = T - (n_lags - 1) = T - l,

    with n_lags = 2N so that the largest lag is l = 2N - 1.

    Every lag is an average of the SAME number of products, T_e, over
    the SAME range of base indices s -- the text's "<.> is the average
    over k = 1, ..., T - l". The largest time index touched is
    T_e - 1 + l = T - 1 and the smallest is 0, so the estimator uses the
    whole record exactly once and no lag is favoured by a longer window.

    (Before 2026-09 each lag was divided by its own window T - k. That
    made the divisor lag-dependent and left the lag sequence without a
    common window, which is what the paper's Eq. (16) prescribes. The
    difference is O(l/T) but it also cost the Toeplitz factors their
    guaranteed positive semi-definiteness.)
    """
    Y = X[:, cols].astype(float)
    Y = Y - Y.mean(axis=0)
    T = Y.shape[0]
    T_eff = T - (n_lags - 1)
    if T_eff < 1:
        raise ValueError(
            f"record too short: n_lags={n_lags} needs T > {n_lags - 1}, "
            f"got T={T}")
    return [(Y[k:k + T_eff].T @ Y[:T_eff]) / T_eff for k in range(n_lags)]


def empirical_moment_matrices(X, cols, N):
    """
    The text's Eq. (16) taken literally: build the stacked past and
    future vectors and return (<ff^T>, <fp^T>, <pp^T>).

    With l = 2N - 1 and T_e = T - l windows k = 0, ..., T_e - 1,

        p_a = y[k + N - 1 - a],   a = 0, ..., N-1   (newest lag first)
        f_b = y[k + N + b],       b = 0, ..., N-1

    so past and future are contiguous and disjoint, and the cross block
    (b, a) has lag (N + b) - (N - 1 - a) = a + b + 1, i.e. exactly the
    Hankel pattern Lambda(a+b+1) of Eq. (11).

    Returns three (N*m, N*m) matrices, each an average over the same
    T_e windows.

    Why this form matters: <ff^T> = F^T F / T_e and <pp^T> = P^T P / T_e
    are Gram matrices of actual stacked data. They are therefore exactly
    positive semi-definite, and positive definite as soon as T_e >= N*m
    with the record in general position. That is what lets the whitening
    be computed exactly, with no eigenvalue floor, and what makes the
    resulting canonical correlations lie in [0, 1] by Cauchy-Schwarz
    rather than by clipping.
    """
    Y = X[:, cols].astype(float)
    Y = Y - Y.mean(axis=0)
    T, m = Y.shape
    l = 2 * N - 1
    T_eff = T - l
    if T_eff < 1:
        raise ValueError(f"record too short (T={T}) for N={N} (l={l})")
    if T_eff <= 2 * N * m:
        warnings.warn(
            f"T_e={T_eff} <= p+q={2 * N * m}: there are no more windows "
            f"than the past and future dimensions together, so some "
            f"canonical correlations are exactly 1 and the rank posterior "
            f"is not defined. Reduce N or use a longer record.",
            RuntimeWarning, stacklevel=2)

    P = np.hstack([Y[N - 1 - a: N - 1 - a + T_eff] for a in range(N)])
    F = np.hstack([Y[N + b: N + b + T_eff] for b in range(N)])
    return (F.T @ F) / T_eff, (F.T @ P) / T_eff, (P.T @ P) / T_eff


# ======================================================================
# 3. Core builders (operate on a lag list, shared by both worlds)
# ======================================================================
def hankel(lams, N):
    """
    N x N block Hankel of Eq. hankel: block (a, b) = Lambda(a + b + 1),
    lags 1..2N-1 (= l). Pass the standard lag list Lambda(0..2N-1);
    Lambda(0) is simply not used here.
    """
    if len(lams) < 2 * N:
        raise ValueError(f"need lags 0..{2*N-1} for N={N}, "
                         f"got {len(lams)}")
    return np.block([[lams[a + b + 1] for b in range(N)] for a in range(N)])


def _toeplitz_pp(lams, N):
    """<pp^T> from a lag list: block (a1,a2) = Lambda(a2-a1), a2 >= a1."""
    return np.block([[lams[a2 - a1] if a2 >= a1 else lams[a1 - a2].T
                      for a2 in range(N)] for a1 in range(N)])


def _toeplitz_ff(lams, N):
    """<ff^T> from a lag list: block (b1,b2) = Lambda(b1-b2), b1 >= b2."""
    return np.block([[lams[b1 - b2] if b1 >= b2 else lams[b2 - b1].T
                      for b2 in range(N)] for b1 in range(N)])


def _inv_sqrt(S, name="matrix"):
    """
    EXACT symmetric inverse square root S^-1/2, no eigenvalue floor.

    S must be positive definite. If it is not, this raises rather than
    regularising: a non-positive eigenvalue in <ff^T> or <pp^T> means
    the record is too short (T_e < N*m) or the stacked past/future is
    collinear, and the canonical correlations computed from a patched
    inverse would be an artefact of the patch, not of the data.

    (Before 2026-09 this clipped eigenvalues at rcond * lam_max with
    rcond = 1e-10. That silently replaced any direction below the floor
    by a bounded weight, so the "W has the same rank as H-hat" argument
    -- which assumes the exact inverse -- did not apply to what was
    actually computed.)
    """
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    if w[0] <= 0.0:
        raise np.linalg.LinAlgError(
            f"{name} is not positive definite (eigenvalues from "
            f"{w[0]:.3e} to {w[-1]:.3e}); its inverse square root does "
            f"not exist. Reduce the block depth N, or use a longer "
            f"record: the past/future stack needs T_e >= N*m for a "
            f"full-rank Gram matrix.")
    cond = w[-1] / w[0]
    if cond > 1.0 / np.finfo(float).eps:
        warnings.warn(
            f"{name} is numerically singular (condition number "
            f"{cond:.3e} > 1/eps): the whitening is exact but the "
            f"smallest directions carry no significant digits, so the "
            f"trailing canonical correlations are not meaningful.",
            RuntimeWarning, stacklevel=3)
    return V @ np.diag(w ** -0.5) @ V.T


def _whiten(C_ff, C_fp, C_pp, method="eigh"):
    """
    W = <ff^T>^-1/2 <fp^T> <pp^T>^-1/2, computed exactly.

    method='eigh'      the symmetric inverse square root written in the
                       text.
    method='cholesky'  W = L_f^-1 <fp^T> L_p^-T with L L^T = C. Gives
                       IDENTICAL singular values -- the two W differ by
                       an orthogonal factor on each side -- at lower
                       cost, and fails cleanly (LinAlgError) when a
                       factor is not positive definite.
    """
    if method == "eigh":
        return _inv_sqrt(C_ff, "<ff^T>") @ C_fp @ _inv_sqrt(C_pp, "<pp^T>")
    if method == "cholesky":
        L_f = np.linalg.cholesky(0.5 * (C_ff + C_ff.T))
        L_p = np.linalg.cholesky(0.5 * (C_pp + C_pp.T))
        return np.linalg.solve(L_f, np.linalg.solve(L_p, C_fp.T).T)
    raise ValueError("method must be 'eigh' or 'cholesky'")


def whitened_hankel(lams, N, method="eigh"):
    """
    W = <ff^T>^-1/2 <fp^T> <pp^T>^-1/2 built from a LAG LIST, so the
    three moment matrices keep their exact block-Hankel / block-Toeplitz
    structure. Needs lags 0..2N-1.

    Singular values are the past-future canonical correlations, so
    rank(H) = #{nonzero}. Whitening leaves the rank unchanged (Sigma,
    hence both Toeplitz factors, is positive definite because D is
    nonsingular) but makes the estimate invariant to the units of the
    sensors, bounds the spectrum in [0,1], and equalizes the otherwise
    strongly anisotropic estimation noise across directions.

    NOTE for the EMPIRICAL case: a Toeplitz matrix assembled from
    estimated lags is not a Gram matrix, so it is not guaranteed
    positive definite and the singular values are not guaranteed to lie
    in [0, 1]. Use empirical_whitened_hankel(..., structured=False),
    the default, when working from a record -- that route builds the
    moment matrices directly from the stacked data and both guarantees
    hold. This structured form is exact for THEORETICAL lags, where
    Sigma is positive definite by construction.
    """
    if len(lams) < 2 * N:
        raise ValueError(f"need lags 0..{2*N-1} for N={N}, "
                         f"got {len(lams)}")
    return _whiten(_toeplitz_ff(lams, N), hankel(lams, N),
                   _toeplitz_pp(lams, N), method=method)


# ======================================================================
# 4. Hankels
# ======================================================================
def theoretical_hankel(sys, sensors, h, N):
    return hankel(theoretical_lags(sys, sensors, h, 2 * N), N)


def empirical_hankel(X, cols, N):
    """Raw (unwhitened) Hankel from constant-window lag estimates."""
    return hankel(empirical_lags(X, cols, 2 * N), N)


def theoretical_whitened_hankel(sys, sensors, h, N, method="eigh"):
    return whitened_hankel(theoretical_lags(sys, sensors, h, 2 * N), N,
                           method=method)


def empirical_whitened_hankel(X, cols, N, structured=False, method="eigh"):
    """
    Whitened Hankel from a record.

    structured=False (default)  moment matrices straight from the
        stacked past/future of Eq. (16). <ff^T> and <pp^T> are Gram
        matrices: exactly PSD, PD when T_e >= N*m, and the singular
        values lie in [0, 1] by Cauchy-Schwarz. No clipping is needed
        anywhere downstream.

    structured=True  assemble block-Toeplitz / block-Hankel matrices
        from the constant-window lag estimates instead. This keeps the
        exact algebraic structure the theory uses, at the price of the
        two guarantees above: the factors may fail to be positive
        definite (then _inv_sqrt raises) and gamma may exceed 1 slightly
        (then the rank posterior raises). Useful as a check that the
        two constructions agree when T is comfortable.
    """
    if structured:
        return whitened_hankel(empirical_lags(X, cols, 2 * N), N,
                               method=method)
    C_ff, C_fp, C_pp = empirical_moment_matrices(X, cols, N)
    return _whiten(C_ff, C_fp, C_pp, method=method)


# ======================================================================
# 5. Spectra
# ======================================================================
def theoretical_singular_values(sys, sensors, h, N):
    """Singular values of the raw (unwhitened) Hankel."""
    return np.linalg.svd(theoretical_hankel(sys, sensors, h, N),
                         compute_uv=False)


def empirical_singular_values(X, cols, N):
    return np.linalg.svd(empirical_hankel(X, cols, N), compute_uv=False)


def theoretical_canonical_correlations(sys, sensors, h, N, method="eigh"):
    """
    Past-future canonical correlations. Returned unmodified -- they lie
    in [0, 1] because the whitening is exact and Sigma is positive
    definite; values outside that range are a numerical warning sign,
    not something to be clipped away.
    """
    return np.linalg.svd(
        theoretical_whitened_hankel(sys, sensors, h, N, method=method),
        compute_uv=False)


def empirical_canonical_correlations(X, cols, N, structured=False,
                                     method="eigh"):
    """
    Sample past-future canonical correlations, returned unmodified.

    With the default structured=False these lie in [0, 1] exactly, by
    Cauchy-Schwarz on the stacked past/future -- no clipping.
    """
    return np.linalg.svd(
        empirical_whitened_hankel(X, cols, N, structured=structured,
                                  method=method),
        compute_uv=False)


# ======================================================================
# 6. Theoretical rank and causality (numerical checks of the Theorem)
# ======================================================================
def theoretical_rank(sys, sensors, h, N, tol=None):
    """
    Rank of the noise-free Hankel; equals n_p by the Theorem, so this is
    the numerical check of the theory. Uses numpy's default tolerance
    (sigma_max * max(dim) * eps) on the RAW Hankel -- no hand-picked
    threshold. A shortfall means double precision has run out (the
    canonical correlations of the missing modes are below ~1e-16), not
    that the theorem failed; compare against the condition number.
    """
    H = theoretical_hankel(sys, sensors, h, N)
    return int(np.linalg.matrix_rank(H) if tol is None
               else np.linalg.matrix_rank(H, tol=tol))


def theoretical_kalman_causality(sys, sensors_i, sensors_j, h, N):
    """
    Noise-free causality verdict:
        rank(H^j) == rank(H^{ij})  <=>  i ~~> j  (i causes j).
    Should agree with the graph, i.e. with the existence of a directed
    path from sensors_i to sensors_j, which makes it the check of the
    reachability corollary. N must satisfy N >= |P_i union P_j|.
    """
    both = list(sensors_j) + [v for v in sensors_i if v not in sensors_j]
    r_j = theoretical_rank(sys, sensors_j, h, N)
    r_ij = theoretical_rank(sys, both, h, N)
    return {"causes": r_j == r_ij, "rank_j": r_j, "rank_ij": r_ij}


# ======================================================================
# 7. Rank posterior and causality log-odds (Estimation section)
# ======================================================================
# Likelihood. Past-future CCA is the Gaussian reduced-rank regression of
# the stacked future on the stacked past: the log-likelihood maximized
# under a rank-r constraint on the past->future regression matrix
# (equal, almost surely, to the maximum under rank <= r) satisfies
#
#     L(r) - L(r_max) = (T_e/2) * sum_{k>r} ln(1 - gamma_k^2),
#
# (Fujikoshi 1974; Anderson 1951), so the increment from r-1 to r is
# (T_e/2) ln(1/(1-gamma_r^2)) -- the paper's Bayes-factor numerator.
#
# Posterior. ln B_r ~ L(r) - L(r-1) - (Delta_dr/2) ln T_e with
# Delta_dr = p + q - 2r + 1 = m(l+1) + 1 - 2r for p = q = m(l+1)/2:
# the increment of d_r = r(p+q-r), the dimension of the manifold of
# p x q rank-r matrices (M = L R^T gives r(p+q) parameters minus the
# r^2-dimensional GL(r) gauge). The posterior over ranks is a softmax
# of the cumulative sums, and the causality probability conditions the
# product of the two posteriors on the cone rho_ij >= rho_j
# (Eq. main:proba); the log-odds is reported.


def rank_posterior_from_gammas(gammas, T_eff, dims=None, prior=None,
                               return_log=False):
    """
    P(rank = r | data) for r = 0..len(gammas), from the canonical
    correlations of a whitened p x q matrix, via BIC weights:

      log w(r) = sum_{k<=r} [ (T_eff/2) * (-log(1 - g_k^2))
                              - ((p + q - 2k + 1)/2) * log(T_eff) ]
                 + log prior(r),
      P(rank = r) = w(r) / sum_r' w(r').

    gammas : canonical correlations, sorted decreasing (as returned by
             *_canonical_correlations). They are VALIDATED, not clipped:
             a value outside [0, 1) raises, because g >= 1 makes the
             likelihood increment infinite and there is no honest
             default to substitute. g = 1 means a direction of the
             future is a perfect linear function of the past, which at
             finite T_e means T_e < p (too few windows) or exact
             collinearity in the record -- both of which the caller
             needs to see.
    T_eff  : effective sample size T_e = T - l (see
             empirical_rank_posterior).
    dims   : (p, q) of the whitened matrix; default (m, m), m = len(g).
    prior  : optional length m+1 array over r = 0..m; need not be
             normalized; zeros allowed (excluded ranks).

    Because the g_k are sorted, the per-component increments decrease,
    so the log-posterior is concave in r and the MAP is unique -- it is
    the BIC rank estimate. Component k contributes weight > 1 iff

        g_k^2 > 1 - T_eff^(-(p+q-2k+1)/T_eff)
              ~ (p + q - 2k + 1) * log(T_eff) / T_eff,

    the finite-sample detection floor: the T*gamma^2 >> 1 rule of thumb
    made quantitative. Modes near the floor get weight ~ 1 and spread
    the posterior over adjacent ranks, which is the honest verdict.
    """
    g = np.asarray(gammas, dtype=float)
    if g.ndim != 1:
        raise ValueError("gammas must be one-dimensional")
    if not np.all(np.isfinite(g)):
        raise ValueError("gammas contains non-finite values")
    if g.min() < 0.0:
        raise ValueError(
            f"negative canonical correlation ({g.min():.3e}); singular "
            f"values cannot be negative, so the whitened matrix was not "
            f"formed correctly")
    if g.max() >= 1.0:
        raise ValueError(
            f"canonical correlation {g.max():.6f} >= 1: the likelihood "
            f"increment -log(1 - g^2) is infinite and the rank posterior "
            f"is undefined. A unit canonical correlation means the "
            f"past/future Gram matrices are rank deficient (need "
            f"T_e >= p) or the record is exactly collinear. Reduce the "
            f"block depth N, or lengthen the record.")
    m = g.size
    p, q = (m, m) if dims is None else dims
    k = np.arange(1, m + 1)
    incr = 0.5 * T_eff * (-np.log1p(-g ** 2)) \
        - 0.5 * (p + q - 2 * k + 1) * np.log(T_eff)
    logw = np.concatenate(([0.0], np.cumsum(incr)))
    if prior is not None:
        prior = np.asarray(prior, dtype=float)
        if prior.size != m + 1:
            raise ValueError(f"prior must have length {m + 1}")
        with np.errstate(divide="ignore"):
            logw = logw + np.log(prior)
    logw -= np.max(logw)
    if return_log:
        return logw - logsumexp(logw)
    w = np.exp(logw)
    return w / w.sum()


def empirical_rank_posterior(X, cols, N, prior=None, exclude_zero=True,
                             return_log=False, structured=False,
                             method="eigh"):
    """
    Posterior over the rank of the sensors' Hankel, from a record.

    T_eff = T - l = T - (2N - 1), the number of complete (future, past)
    windows in the record -- the text's T_e, and now also the exact
    number of terms averaged into every entry of every moment matrix
    (crude but adequate; a HAC/long-run-variance refinement would make
    T_eff mode-dependent).

    exclude_zero=True puts prior mass 0 on rank 0: by the Theorem the
    true rank is >= 1 (every sensor lies in its own reachable set, since
    zero-length paths are allowed), so rank 0 is impossible and should
    not absorb posterior mass when the record is short or h too coarse
    for the node's own timescale. The prior is uniform over the
    remaining ranks.

    NOTE that this makes the prior uniform on r >= 1, NOT "uniform and
    strictly positive"; and the support differs between the two Hankels
    a causality test compares (0..N*m_j against 0..N*m_ij). Both facts
    belong in the Methods text.
    """
    T = X.shape[0]
    T_eff = T - (2 * N - 1)
    if T_eff < 2:
        raise ValueError(f"record too short (T={T}) for N={N}")
    m = len(cols)
    p = N * m
    if T_eff <= 2 * p:
        # CCA between a p-dimensional past and a q = p dimensional future
        # needs strictly more windows than p + q, otherwise the leading
        # canonical correlations are exactly 1 by construction.
        # T_e = T - (2N - 1) > 2Nm  <=>  N < (T + 1) / (2(m + 1))
        n_max = max(1, int(np.ceil((T + 1) / (2 * (m + 1)))) - 1)
        raise ValueError(
            f"T_e = {T_eff} <= p + q = {2 * p}: there are no more "
            f"(past, future) windows than the two stacks have dimensions, "
            f"so the leading canonical correlations are exactly 1 by "
            f"construction and the rank posterior is undefined. With T={T} "
            f"and m={m} column(s) the block depth must satisfy "
            f"N <= {n_max} (currently N={N}); note the binding case is the "
            f"JOINT Hankel, which has m = m_i + m_j columns.")
    g = empirical_canonical_correlations(X, cols, N, structured=structured,
                                         method=method)
    m = g.size
    pr = np.ones(m + 1)
    if exclude_zero:
        pr[0] = 0.0
    if prior is not None:
        pr = pr * np.asarray(prior, dtype=float)
    return rank_posterior_from_gammas(g, T_eff, dims=(p, p), prior=pr,
                                      return_log=return_log)


def kalman_causality_probability_from_posteriors(post_j, post_ij):
    """
    P(i causes j) from P_j = P(rank H^j = .) and P_ij = P(rank H^ij = .),
    Eq. main:proba of the text.

    Causality <=> rank(H^j) == rank(H^ij) (reachability corollary), and
    the reachable sets nest, so the pair (r_j, r_ij) lives on the cone
    r_ij >= r_j. The two posteriors come from the same record, so they
    are treated as independent APART from this hard constraint: the
    product measure is conditioned on the cone,

        P(cause) = sum_r P_j(r) P_ij(r)
                   / sum_{r <= r'} P_j(r) P_ij(r').

    Numerator <= denominator always, so the result is in [0, 1]. Exact
    in the concentration limit (both posteriors -> point masses, where
    it reduces to the rank equality); an approximation at finite T,
    validated by ROC calibration. If the feasible mass is 0 the data
    contradict r_ij >= r_j outright and 0.0 is returned.
    """
    a = np.asarray(post_j, dtype=float)
    b = np.asarray(post_ij, dtype=float)
    L = max(a.size, b.size)
    a = np.pad(a, (0, L - a.size))
    b = np.pad(b, (0, L - b.size))
    num = float(np.sum(a * b))
    tail_b = np.concatenate(([1.0], 1.0 - np.cumsum(b)[:-1]))  # P(r_ij >= r)
    den = float(np.sum(a * tail_b))
    return num / den if den > 0.0 else 0.0


def kalman_causality_log_odds_from_posteriors(logp_j, logp_ij):
    """
    Log-odds of causality, log P(cause) - log(1 - P(cause)), from LOG
    rank posteriors, computed entirely in log space (logsumexp).

    Same cone construction as the probability version: P(cause) is the
    diagonal mass r_j = r_ij, its complement the strict cone
    r_ij > r_j -- so the text's "P(rho_ij != rho_j)" is really
    P(rho_ij > rho_j), the sub-cone r_ij < r_j having been removed by
    the conditioning rather than by the posteriors. Working in log space
    matters because the probabilities saturate: evidence differences of
    hundreds of nats collapse to 0.0 or 1.0 in floating point, while the
    log-odds stay resolved. Returns +inf / -inf when one side has no
    support at all.
    """
    a = np.asarray(logp_j, dtype=float)
    b = np.asarray(logp_ij, dtype=float)
    L = max(a.size, b.size)
    a = np.pad(a, (0, L - a.size), constant_values=-np.inf)
    b = np.pad(b, (0, L - b.size), constant_values=-np.inf)
    with np.errstate(invalid="ignore"):
        log_num = logsumexp(a + b)                      # r_ij = r_j
        off = [a[r] + logsumexp(b[r + 1:]) for r in range(L - 1)
               if np.isfinite(a[r])]                    # r_ij > r_j
    log_off = logsumexp(off) if off else -np.inf
    if log_num == -np.inf:
        return -np.inf
    if log_off == -np.inf:
        return np.inf
    return float(log_num - log_off)


def empirical_kalman_causality_probability(X, cols_i, cols_j, N,
                                           prior_j=None, prior_ij=None,
                                           structured=False, method="eigh"):
    """
    Empirical probability that sensor set i causes sensor set j, i.e.
    that a directed path runs from the i-nodes to the j-nodes
    (i ~~> j), following the text's convention rho_ij vs rho_j.

    Arguments are COLUMN indices into X; the same block depth N is used
    for both Hankels. Returns a dict:
      prob          P(i causes j | record), in [0, 1]
      log_odds      ln P(cause) - ln P(not cause)
      posterior_j   P(rank H^j = r),  r = 0..N*M_j
      posterior_ij  P(rank H^ij = r), r = 0..N*M_ij
      map_rank_j, map_rank_ij   the MAP (= BIC) rank estimates

    Reliability is asymmetric: prob near 1 is trustworthy (under
    non-causality, i's own state is directly visible in H^ij and
    separates the posteriors), while prob near 0 only excludes paths
    visible above the detection floor.

    The binding sample-size requirement comes from the JOINT Hankel:
    T_e = T - (2N - 1) must be at least N * len(cols_ij). With one
    column per node that is T >= 2N^2 + 2N - 1.
    """
    cols_ij = list(cols_j) + [c for c in cols_i if c not in cols_j]
    logp_j = empirical_rank_posterior(X, cols_j, N, prior=prior_j,
                                      return_log=True, structured=structured,
                                      method=method)
    logp_ij = empirical_rank_posterior(X, cols_ij, N, prior=prior_ij,
                                       return_log=True, structured=structured,
                                       method=method)
    log_odds = kalman_causality_log_odds_from_posteriors(logp_j, logp_ij)
    return {"prob": float(expit(log_odds)),
            "log_odds": log_odds,
            "posterior_j": np.exp(logp_j),
            "posterior_ij": np.exp(logp_ij),
            "map_rank_j": int(np.argmax(logp_j)),
            "map_rank_ij": int(np.argmax(logp_ij))}

