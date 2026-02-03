import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax
import jax.scipy.linalg as jsl
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
import matplotlib.animation as animation
from typing import Union, Tuple, Iterator
import fractions
import itertools
import functools
import math
import numpy as np
from diffrax import diffeqsolve, Dopri8, Tsit5, ODETerm, SaveAt, PIDController
import diffrax
import optimistix as optx
import optax

import jax
import jax.numpy as jnp
from jax.experimental import jet
from jax.flatten_util import ravel_pytree
import qutip as qt
from scipy.integrate import solve_ivp
import time

jax.config.update("jax_enable_x64", True)

Rational = Union[int, fractions.Fraction]

# --- [Existing Math Helper Functions Remain Unchanged] ---

@functools.lru_cache(maxsize=None)
def binomial_coeff(n: Rational, k: int) -> Rational:
    if k < 0:
        raise ValueError("k must be a non-negative integer")
    n = fractions.Fraction(n)
    result = 1
    for i in range(k):
        result *= (n - i) / (i + 1)
    return result

MultiIndex = Tuple[int, ...]

def multi_index_lte(i: MultiIndex) -> Iterator[MultiIndex]:
    return itertools.islice(
        itertools.product(*(range(0, k + 1) for k in i)),
        1,
        None,
    )

@functools.lru_cache(maxsize=None)
def multi_index_binomial_coeff(i: Tuple[Rational, ...],
                               j: MultiIndex) -> Rational:
    result = 1
    for i_k, j_k in zip(i, j):
        result *= binomial_coeff(i_k, j_k)
    return result

def coeffs(i: MultiIndex) -> Iterator[Tuple[MultiIndex, Rational]]:
    for j in multi_index_lte(i):
        c = multi_index_binomial_coeff(i, j) * (-1)**(sum(i) - sum(j))
        if c != 0:
            yield j, c

@functools.lru_cache(maxsize=None)
def precompute_kc(orders: MultiIndex) -> Tuple[np.ndarray, np.ndarray]:
    ks, cs = [], []
    for k, c in coeffs(orders):
        k = np.array(list(map(float, k)))
        c = float(c)
        ks.append(list(k))
        cs.append(c)
    ks = np.array(ks)
    cs = np.array(cs)
    return ks, cs

@functools.lru_cache(maxsize=None)
def structured_orders_to_flat(
    orders: Tuple[Tuple[int, int, int], ...],
    num_modes: int
) -> MultiIndex:
    """
    Convert [(site, phase, multiplicity), ...] to flat multi-index.
    """
    D = num_modes * 4
    flat = [0] * D

    for site, phase, mult in orders:
        if mult == 0:
            continue
        if site < 0 or site >= num_modes:
            raise ValueError(f"Invalid site index {site}")
        if phase < 0 or phase >= 4:
            raise ValueError(f"Invalid phase index {phase}")
        if mult < 0:
            raise ValueError("Multiplicity must be non-negative")

        if phase < 2:
            flat[site * 2 + phase] += mult
        else:
            flat[2 * num_modes + site * 2 + (phase - 2)] += mult

    return tuple(flat)


def mixed_partial(f, orders: MultiIndex):
    """
    Compute mixed partial derivative specified by flat multi-index `orders`
    using JAX jets. Works for arbitrary dimension.
    """
    d = sum(orders)

    if d == 0:
        return lambda x0: f(x0)
    
    ks, cs = precompute_kc(orders)
    dfact = math.factorial(d)

    def partial_der(x0):
        x0 = jnp.asarray(x0)

        @jax.vmap
        def jet_eval(k):
            hs = (k,) + (jnp.zeros_like(x0),) * (d - 1)
            _, (*_, res) = jet.jet(f, (x0,), (hs,))
            return res

        return jnp.sum(jet_eval(ks) * cs) / dfact

    return partial_der

# --- [Physical Constants and Model Definition] ---

S = 8
G = 0         # Two-photon drive strength
delta = 1.0        # Detuning
U = 10           # Kerr nonlinearity
gamma = 1.0       # Single-photon loss rate
J = 10           # Hopping strength

def calculate_covariance(covariance_params):
    params_reshaped = covariance_params.reshape(-1, 2)
    
    def single_mode_cov(p):
        r, phi = p
        S = jnp.array([
            [jnp.cos(phi), jnp.sin(phi)],
            [jnp.sin(phi), -jnp.cos(phi)]
        ])
        return 0.5*jnp.cosh(2*r)*jnp.identity(2)-0.5*jnp.sinh(2*r)*S

    sigmas = jax.vmap(single_mode_cov)(params_reshaped)
    return jsl.block_diag(*sigmas)


def covariance_sum_inv(covariance_params_a, covariance_params_b):
    params_a = covariance_params_a.reshape(-1, 2)
    params_b = covariance_params_b.reshape(-1, 2)
    
    def single_mode_inv(pa, pb):
        ra, phia = pa
        rb, phib = pb
        
        denom = (1. + jnp.cosh(ra)**2 * jnp.cosh(2 * rb) + 
                 jnp.cosh(2 * rb) * jnp.sinh(ra)**2 - 
                 4. * jnp.cos(phia - phib) * jnp.cosh(ra) * jnp.cosh(rb) * jnp.sinh(ra) * jnp.sinh(rb))
                 
        num00 = (jnp.cosh(2 * ra) + jnp.cosh(2 * rb) + 
                 2. * jnp.cos(phia) * jnp.cosh(ra) * jnp.sinh(ra) + 
                 2. * jnp.cos(phib) * jnp.cosh(rb) * jnp.sinh(rb))
                 
        num01 = (2. * jnp.cosh(ra) * jnp.sin(phia) * jnp.sinh(ra) + 
                 2. * jnp.cosh(rb) * jnp.sin(phib) * jnp.sinh(rb))
                 
        num11 = (jnp.cosh(2 * ra) + jnp.cosh(2 * rb) - 
                 2. * jnp.cos(phia) * jnp.cosh(ra) * jnp.sinh(ra) - 
                 2. * jnp.cos(phib) * jnp.cosh(rb) * jnp.sinh(rb))
                 
        return jnp.array([[num00, num01], [num01, num11]]) / denom

    invs = jax.vmap(single_mode_inv)(params_a, params_b)
    return jsl.block_diag(*invs)


def gen_func(gaussian_a, gaussian_b, Js):
    normalization_a, param_a = gaussian_a
    normalization_b, param_b = gaussian_b
    mean_a, covariance_params_a = unwrap_params(param_a)
    mean_b, covariance_params_b = unwrap_params(param_b)
    alpha_a, beta_a = mean_a[:, ::2].flatten(), mean_a[:, 1::2].flatten()
    alpha_b, beta_b =  mean_b[:, ::2].flatten(), mean_b[:, 1::2].flatten()
    J, J_tilde = Js[:Js.size//2], Js[Js.size//2:]
    
    num_modes = param_a.shape[0]
    covariance_a = calculate_covariance(covariance_params_a)
    covariance_b = calculate_covariance(covariance_params_b)
    covariance_sum = covariance_a + covariance_b
    inv_sigma_sum = covariance_sum_inv(covariance_params_a, covariance_params_b)
    v1 = alpha_b - alpha_a - jnp.dot(covariance_a, J) - J_tilde
    inv_sigma_sum_v1 = jnp.dot(inv_sigma_sum, v1)
    
    diff_beta = beta_b - beta_a
    sum_beta = beta_b + beta_a
    inv_sigma_sum_diff_beta = jnp.dot(inv_sigma_sum, diff_beta)
    inv_sigma_sum_sum_beta = jnp.dot(inv_sigma_sum, sum_beta)

    Z1 = jnp.exp(-0.5 * jnp.dot(v1, inv_sigma_sum_v1))
    M_minus = jnp.exp(0.5 * jnp.dot(diff_beta, inv_sigma_sum_diff_beta))
    M_plus = jnp.exp(0.5 * jnp.dot(sum_beta, inv_sigma_sum_sum_beta))
    
    J_dot_beta = jnp.dot(J, beta_a)
    mix_term_minus = jnp.dot(diff_beta, inv_sigma_sum_v1)
    mix_term_plus = jnp.dot(sum_beta, inv_sigma_sum_v1)
    
    C_minus = jnp.cos(J_dot_beta - mix_term_minus)
    C_plus = jnp.cos(J_dot_beta + mix_term_plus)
    num_exponent = 0.5 * jnp.dot(J, jnp.dot(covariance_a, J)) + jnp.dot(J, alpha_a)
    
    sign, logdet = jnp.linalg.slogdet(covariance_sum)
    log_denominator = 0.5 * (2*num_modes * jnp.log(2 * jnp.pi) + logdet)
    prefactor = 0.5 * jnp.exp(num_exponent - log_denominator)
    
    return normalization_a*normalization_b*prefactor * Z1 * (M_minus * C_minus + M_plus * C_plus)


def get_gen_func_components(gaussian_a, gaussian_b):
    """
    Computes the scalar arguments of the generating function (A13) at J=0.
    These represent the full system's contribution to the exponents and phases.
    Returns the raw dot-product values, not the final Z.
    """
    normalization_a, param_a = gaussian_a
    normalization_b, param_b = gaussian_b
    
    # Unwrap parameters
    mean_a, covariance_params_a = unwrap_params(param_a)
    mean_b, covariance_params_b = unwrap_params(param_b)
    alpha_a, beta_a = mean_a[:, ::2].flatten(), mean_a[:, 1::2].flatten()
    alpha_b, beta_b = mean_b[:, ::2].flatten(), mean_b[:, 1::2].flatten()
    
    # Block diagonal covariances allow simple summation of determinants
    num_modes = param_a.shape[0]
    covariance_a = calculate_covariance(covariance_params_a) # (2M, 2M) but block diag
    covariance_b = calculate_covariance(covariance_params_b)
    covariance_sum = covariance_a + covariance_b
    
    # We need the inverse of the sum. For block diagonal, this is efficient.
    # Note: For optimization, you could sum the inverses of 2x2 blocks, 
    # but using jax.scipy.linalg for the full block-diag matrix is okay for now.
    inv_sigma_sum = covariance_sum_inv(covariance_params_a, covariance_params_b)

    # --- Compute Vectors at J=0 ---
    # v1 at J=0 is just (alpha' - alpha)
    v1_0 = alpha_b - alpha_a 
    diff_beta = beta_b - beta_a
    sum_beta = beta_b + beta_a
    
    # --- Compute Dot Products (Scalars) ---
    # Arg Z1[cite: 590]: -0.5 * v1^T Sigma_sum^-1 v1
    # Note: (A14) defines Z1 with the negative sign in the exp.
    arg_Z1_0 = -0.5 * jnp.dot(v1_0, jnp.dot(inv_sigma_sum, v1_0))
    
    # Arg M +/-[cite: 593, 595]: 0.5 * beta_diff^T Sigma_sum^-1 beta_diff
    arg_M_minus = 0.5 * jnp.dot(diff_beta, jnp.dot(inv_sigma_sum, diff_beta))
    arg_M_plus  = 0.5 * jnp.dot(sum_beta,  jnp.dot(inv_sigma_sum, sum_beta))
    
    # Arg C +/-: (beta +/- beta)^T Sigma_sum^-1 (alpha' - alpha)
    # Note: J=0 removes the J^T terms.
    inv_sigma_v1_0 = jnp.dot(inv_sigma_sum, v1_0)
    arg_C_minus_0 = -jnp.dot(diff_beta, inv_sigma_v1_0) # First term J^T beta is 0
    arg_C_plus_0  =  jnp.dot(sum_beta,  inv_sigma_v1_0)
    
    # Prefactor (Normalization + Determinant)
    sign, logdet = jnp.linalg.slogdet(covariance_sum)
    log_denominator = 0.5 * (2 * num_modes * jnp.log(2 * jnp.pi) + logdet)
    # The prefactor exp also has a J term (num_exponent), at J=0 it is 0.
    log_prefactor = -log_denominator
    
    base_prefactor = 0.5*normalization_a * normalization_b * jnp.exp(log_prefactor)
    
    return base_prefactor, arg_Z1_0, arg_M_minus, arg_M_plus, arg_C_minus_0, arg_C_plus_0


def gen_func_local_correct(gaussian_a_slice, gaussian_b_slice, global_components, Js_slice):
    """
    Computes Eq (A13) by adding local J-dependent deviations to the global baseline.
    Only involves dot products of size (2*k), where k is number of active modes.
    """
    # Unpack Global Baselines (Constants wrt J)
    base_prefactor, arg_Z1_glob, arg_M_minus_glob, arg_M_plus_glob, arg_C_minus_glob, arg_C_plus_glob = global_components
    
    # Unpack Local Slice Params
    norm_a, param_a = gaussian_a_slice # norm is dummy 1.0
    norm_b, param_b = gaussian_b_slice
    
    mean_a, cov_p_a = unwrap_params(param_a)
    mean_b, cov_p_b = unwrap_params(param_b)
    
    # Flatten local arrays
    alpha_a, beta_a = mean_a[:, ::2].flatten(), mean_a[:, 1::2].flatten()
    alpha_b, beta_b = mean_b[:, ::2].flatten(), mean_b[:, 1::2].flatten()
    
    # Local Matrices
    cov_a = calculate_covariance(cov_p_a)
    inv_sigma_sum = covariance_sum_inv(cov_p_a, cov_p_b) # Local inverse
    
    J, J_tilde = Js_slice[:Js_slice.size//2], Js_slice[Js_slice.size//2:]
    
    # --- 1. Compute Local Args at J=0 (To Subtract) ---
    v1_0 = alpha_b - alpha_a
    diff_beta = beta_b - beta_a
    sum_beta = beta_b + beta_a
    
    inv_sigma_v1_0 = jnp.dot(inv_sigma_sum, v1_0)
    
    local_arg_Z1_0 = -0.5 * jnp.dot(v1_0, inv_sigma_v1_0)
    # M terms do not depend on J, so Global M is exact. No deviation needed.
    local_arg_C_minus_0 = -jnp.dot(diff_beta, inv_sigma_v1_0)
    local_arg_C_plus_0  =  jnp.dot(sum_beta,  inv_sigma_v1_0)

    # --- 2. Compute Local Args at J (To Add) ---
    # v1(J) = (alpha' - alpha) - Cov_a J - J_tilde [cite: 574]
    v1_J = v1_0 - jnp.dot(cov_a, J) - J_tilde
    
    inv_sigma_v1_J = jnp.dot(inv_sigma_sum, v1_J)
    
    local_arg_Z1_J = -0.5 * jnp.dot(v1_J, inv_sigma_v1_J)
    
    # C args: J^T beta +/- beta_terms 
    J_dot_beta = jnp.dot(J, beta_a)
    mix_minus = jnp.dot(diff_beta, inv_sigma_v1_J)
    mix_plus  = jnp.dot(sum_beta,  inv_sigma_v1_J)
    
    local_arg_C_minus_J = J_dot_beta - mix_minus
    local_arg_C_plus_J  = J_dot_beta + mix_plus
    
    # --- 3. Compute Prefactor Deviation ---
    # num_exponent 
    local_num_exponent = 0.5 * jnp.dot(J, jnp.dot(cov_a, J)) + jnp.dot(J, alpha_a)

    # --- 4. Reconstruct Total Args ---
    # Effective = Global + (Local_J - Local_0)
    
    eff_arg_Z1 = arg_Z1_glob + (local_arg_Z1_J - local_arg_Z1_0)
    
    # M terms are constant in J, so just use global
    eff_arg_M_minus = arg_M_minus_glob
    eff_arg_M_plus  = arg_M_plus_glob
    
    eff_arg_C_minus = arg_C_minus_glob + (local_arg_C_minus_J - local_arg_C_minus_0)
    eff_arg_C_plus  = arg_C_plus_glob  + (local_arg_C_plus_J  - local_arg_C_plus_0)
    
    # Prefactor deviation
    eff_prefactor = base_prefactor * jnp.exp(local_num_exponent)
    
    # --- 5. Final Assembly (A13) ---
    Z1 = jnp.exp(eff_arg_Z1)
    M_minus = jnp.exp(eff_arg_M_minus)
    M_plus  = jnp.exp(eff_arg_M_plus)
    
    return eff_prefactor * Z1 * (M_minus * jnp.cos(eff_arg_C_minus) + 
                                 M_plus  * jnp.cos(eff_arg_C_plus))

def compute_local_derivative(gaussian_a, gaussian_b, global_components, site_idx, orders_list):
    """
    Applies derivatives specified in orders_list to a single site 'site_idx'.
    orders_list: List of tuples (0, phase, power). Note site is always 0 relative to the slice.
    """
    norm_a, param_a = gaussian_a
    norm_b, param_b = gaussian_b
    
    # Slice 1 mode
    p_a_slice = jax.lax.dynamic_slice(param_a, (site_idx, 0), (1, 6))
    p_b_slice = jax.lax.dynamic_slice(param_b, (site_idx, 0), (1, 6))
    
    g_a_slice = (1.0, p_a_slice)
    g_b_slice = (1.0, p_b_slice)
    
    def local_wrapper(Js_slice):
        return gen_func_local_correct(g_a_slice, g_b_slice, global_components, Js_slice)
    
    # Convert structured orders to flat index for 1 mode (dim=4)
    flat_orders = structured_orders_to_flat(tuple(orders_list), num_modes=1)
    
    return mixed_partial(local_wrapper, flat_orders)(jnp.zeros(4))

def single_photon_drive_optimized(gaussian_a, gaussian_b, global_components):
    num_modes = gaussian_a[1].shape[0]
    
    def site_term(i):
        term1 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,3,1)])
        term2 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,2,1)])
        return jnp.real(S)/jnp.sqrt(2) * term1 - jnp.imag(S)/jnp.sqrt(2) * term2

    return site_term(0)

def double_photon_drive_optimized(gaussian_a, gaussian_b, global_components):
    num_modes = gaussian_a[1].shape[0]
    
    def site_term(i):
        # G * (a^dag^2) terms
        t1 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,1), (0,2,1)])
        t2 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,3,1)])
        
        # G_conj * (a^2) terms
        t3 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,2,1)])
        t4 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,1), (0,3,1)])
        
        return jnp.real(G) * (t1 + t2) - jnp.imag(G) * (t3 - t4)

    return site_term(0)

def delta_term_optimized(gaussian_a, gaussian_b, global_components):
    num_modes = gaussian_a[1].shape[0]
    
    def site_term(i):
        t1 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,1), (0,2,1)])
        t2 = compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,3,1)])
        return t1 - t2

    return delta * jnp.sum(jax.vmap(site_term)(jnp.arange(num_modes)))

def u_term_optimized(gaussian_a, gaussian_b, global_components):
    num_modes = gaussian_a[1].shape[0]
    
    def site_term(i):
        total = 0
        total += compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,3), (0,3,1)])
        total += compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,1,2), (0,3,1)])
        total += -compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,2), (0,1,1), (0,2,1)])
        total += -compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,3), (0,2,1)])
        total += 2 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,1), (0,2,1)])
        total += -2 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,3,1)])
        total += 0.25 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,1), (0,2,3)])
        total += -0.25 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,3,3)])
        total += -0.25 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,2,2), (0,3,1)])
        total += 0.25 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,1), (0,2,1), (0,3,2)])
        return total

    return U/2 * jnp.sum(jax.vmap(site_term)(jnp.arange(num_modes)))

def single_photon_loss_term_optimized(gaussian_a, gaussian_b, global_components):
    num_modes = gaussian_a[1].shape[0]
    
    def site_term(i):
        total = 0
        total += compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,0,1), (0,2,1)])
        total += compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,1,1), (0,3,1)])
        # For the constant term '2*W', we calculate the 0-th order deriv which is just the function value
        total += 2 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [])
        total += 0.5 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,2,2)])
        total += 0.5 * compute_local_derivative(gaussian_a, gaussian_b, global_components, i, [(0,3,2)])    
        return total

    return gamma / 2 * jnp.sum(jax.vmap(site_term)(jnp.arange(num_modes)))

def all_terms_optimized(gaussians_a, gaussians_b):
    norms_a, params_a = gaussians_a
    norms_b, params_b = gaussians_b
    
    def pair_func(gaussian_a, gaussian_b):
        # 1. Compute Global Overlap Scalars (O(N) cost, but no differentiation)
        global_components = get_gen_func_components(gaussian_a, gaussian_b)
        
        # 2. Compute sums of O(1) local derivatives
        val = 0
        val += double_photon_drive_optimized(gaussian_a, gaussian_b, global_components)
        val += delta_term_optimized(gaussian_a, gaussian_b, global_components)
        val += u_term_optimized(gaussian_a, gaussian_b, global_components)
        val += single_photon_loss_term_optimized(gaussian_a, gaussian_b, global_components)
        val += single_photon_drive_optimized(gaussian_a, gaussian_b, global_components)
        val += hopping_term_optimized(gaussian_a, gaussian_b) # Use the version from previous answer
        return val

    # Vmap over the NxN pairs
    matrix_elements = jax.vmap(lambda g_a: jax.vmap(lambda g_b: pair_func(g_a, g_b))(gaussians_b))(gaussians_a)
    return jnp.sum(matrix_elements)


def hopping_term_optimized(gaussian_a, gaussian_b):
    num_modes = gaussian_a[1].shape[0]

    # 1. Compute Global Scalars (O(N)) - No Derivatives here!
    global_components = get_gen_func_components(gaussian_a, gaussian_b)
    
    def site_term(i):
        # 2. Slice Params (O(1))
        norm_a, param_a = gaussian_a
        norm_b, param_b = gaussian_b
        
        # Take modes i and i+1
        p_a_slice = jax.lax.dynamic_slice(param_a, (i, 0), (2, 6))
        p_b_slice = jax.lax.dynamic_slice(param_b, (i, 0), (2, 6))
        
        g_a_slice = (1.0, p_a_slice) # Dummy norm
        g_b_slice = (1.0, p_b_slice)
        
        # 3. Define function to differentiate
        def local_wrapper(Js_slice):
            return gen_func_local_correct(g_a_slice, g_b_slice, global_components, Js_slice)

        # 4. Compute mixed partials using only 2 modes! (O(1))
        # Note: orders must refer to indices 0 and 1 (relative to slice)
        # Term: a_i^dag a_{i+1} -> partials on slice indices 0 and 1
        
        # Orders for -J * (a_i^dag a_{i+1} + a_{i+1}^dag a_i)
        # Indices in slice: i -> 0, i+1 -> 1
        
        # a_0^dag a_1: (0,0,1), (1,3,1)
        term1 = mixed_partial(local_wrapper, structured_orders_to_flat(((0,0,1), (1,3,1)), 2))(jnp.zeros(8))
        
        # a_1^dag a_0: (1,0,1), (0,3,1)
        term2 = mixed_partial(local_wrapper, structured_orders_to_flat(((1,0,1), (0,3,1)), 2))(jnp.zeros(8))
        
        # - a_0 a_1^dag (from commutation? check your old_hopping_term logic)
        # Your old code had: -gen_func... [(i,1,1), (i+1,2,1)]
        term3 = mixed_partial(local_wrapper, structured_orders_to_flat(((0,1,1), (1,2,1)), 2))(jnp.zeros(8))
        
        term4 = mixed_partial(local_wrapper, structured_orders_to_flat(((1,1,1), (0,2,1)), 2))(jnp.zeros(8))
        
        return -J * (term1 + term2 - term3 - term4)

    return jnp.sum(jax.vmap(site_term)(jnp.arange(num_modes - 1)))

def all_terms_optimized(gaussians_a, gaussians_b):
    norms_a, params_a = gaussians_a
    norms_b, params_b = gaussians_b
    
    def pair_func(gaussian_a, gaussian_b):
        # 1. Compute Global Overlap Scalars (O(N) cost, but no differentiation)
        global_components = get_gen_func_components(gaussian_a, gaussian_b)
        # 2. Compute sums of O(1) local derivatives
        val = 0
        val += double_photon_drive_optimized(gaussian_a, gaussian_b, global_components)
        val += delta_term_optimized(gaussian_a, gaussian_b, global_components)
        val += u_term_optimized(gaussian_a, gaussian_b, global_components)
        val += single_photon_loss_term_optimized(gaussian_a, gaussian_b, global_components)
        val += single_photon_drive_optimized(gaussian_a, gaussian_b, global_components)
        val += hopping_term_optimized(gaussian_a, gaussian_b) # Use the version from previous answer
        return val

    # Vmap over the NxN pairs
    matrix_elements = jax.vmap(lambda g_a: jax.vmap(lambda g_b: pair_func(g_a, g_b))(gaussians_b))(gaussians_a)
    return jnp.sum(matrix_elements)

def liouvillian_gradient(gaussians):
    return jax.grad(all_terms_optimized, argnums=0)(gaussians, gaussians)

def geometric_tensor(gaussians):
    norms, params = gaussians
    num_modes = params.shape[1]
    N = norms.shape[0]
    dim_p = num_modes * 6
    def total_gen_func(gaussians_a, gaussians_b):
        return jnp.sum(jax.vmap(lambda g_a: jax.vmap(lambda g_b: gen_func(g_a, g_b, jnp.zeros(4*num_modes)))(gaussians_a))(gaussians_b))
    (T_nn, T_np), (T_pn, T_pp)= jax.jacfwd(jax.grad(total_gen_func, argnums=(1)), argnums=(0))(gaussians, gaussians)
    
    T_np_mat = T_np.reshape(N, N * dim_p)
    T_pn_mat = T_pn.reshape(N * dim_p, N)
    T_pp_mat = T_pp.reshape(N * dim_p, N * dim_p)
    
    T_top = jnp.hstack([T_nn, T_np_mat])
    T_bot = jnp.hstack([T_pn_mat, T_pp_mat])
    T_mat = jnp.vstack([T_top, T_bot])
    return T_mat

def test_geometric_tensor(gaussians):
    """
    Computes the Geometric Tensor using serialized Forward-over-Reverse AD.
    Memory usage: Constant (does not scale with number of parameters).
    """
    # 1. Flatten parameters to treat them as a single vector
    flat_params, unravel_fn = ravel_pytree(gaussians)
    num_params = flat_params.size
    
    # 2. Define the function: Bra_Params -> Gradient_wrt_Ket_Params
    # We want the Jacobian of THIS function.
    def gradient_wrt_ket(bra_params_flat):
        bra_pytree = unravel_fn(bra_params_flat)
        
        # We differentiate total_gen_func wrt the Ket (arg1), holding Bra (arg0) fixed
        # But arg0 is the input to this wrapper 'gradient_wrt_ket'
        def scalar_overlap(ket_pytree):
            # Recalculate total_gen_func sum logic here to ensure scoping
            # Note: We must replicate the summation logic from your original function
            return jnp.sum(jax.vmap(lambda g_a: jax.vmap(lambda g_b: gen_func(g_a, g_b, jnp.zeros(4*num_modes)))(bra_pytree))(ket_pytree))

        # Compute gradient wrt Ket (arg1)
        # This returns a PyTree of size P
        grad_pytree = jax.grad(scalar_overlap)(gaussians) 
        
        # Flatten output to vector
        return ravel_pytree(grad_pytree)[0]

    # 3. Define the Matrix-Vector Product (One Column of T)
    # T @ v = JVP( gradient_wrt_ket, v )
    def get_column(v):
        # jvp performs Forward-Mode AD: efficient and no extra tape overhead
        _, col = jax.jvp(gradient_wrt_ket, (flat_params,), (v,))
        return col

    # 4. Serialize execution with lax.map
    # Instead of computing all 1460 columns at once, we do them one by one.
    # This prevents the memory explosion.
    basis = jnp.eye(num_params)
    T_flat = jax.lax.map(get_column, basis)
    
    return T_flat

def renormalize(normalizations):
    return normalizations/jnp.sum(normalizations)

def number_operator(gaussians, mode):
    normalizations, params = gaussians
    normalizations = renormalize(normalizations)
    def single_param_n(gaussian):
        normalization, param = gaussian
        mean, covariance_params = unwrap_params(param)
        covariance = calculate_covariance(covariance_params[mode])
        mean = mean[mode,::2]+ 1j* mean[mode,1::2]
        return normalization*(jnp.sum(jnp.real(mean**2))+covariance[0,0]+covariance[1,1]-1)/2
    
    return jnp.sum(jax.vmap(single_param_n)((normalizations, params)))

def parity_operator(gaussians, mode):
    normalizations, params = gaussians
    normalizations = renormalize(normalizations)
    def single_param_parity(gaussian):
        normalization, param = gaussian
        mean, covariance_params = unwrap_params(param)
        covariance = calculate_covariance(covariance_params[mode])
        mean = mean[mode,::2]+ 1j* mean[mode,1::2]
        term = jnp.exp(-0.5 * jnp.dot(-mean, jnp.dot(jla.inv(covariance), -mean)))
        return jnp.pi * normalization * jnp.real(term) / jnp.sqrt((2*jnp.pi)**2 * jnp.linalg.det(covariance))

    return jnp.sum(jax.vmap(single_param_parity)((normalizations, params)))

def unwrap_params(params):
    means = params[:, :4]
    covariances = params[:, 4:]
    return means, covariances

def initialize_vacuum_state(N_G, num_modes=1):
    params = jnp.zeros((N_G, num_modes, 6))
    normalizations = jnp.ones(N_G)/N_G
    return normalizations, params

def expand_state_cluster(gaussians, expansion_factor=4, noise_scale=1e-4, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    normalizations, params = gaussians
    num_modes = params.shape[1] 
    old_N = normalizations.shape[0]
    new_N = old_N * expansion_factor
    new_params = jnp.zeros((new_N, num_modes, 6))
    new_normalizations = jnp.zeros(new_N)
    print(f"Expanding ansatz from {old_N} to {new_N} Gaussians...")
    for i in range(old_N):
        base_weight = normalizations[i]
        base_center = params[i, :, :4] # Fixed slicing index from original code
        base_cov = params[i, :, 4:]
        key, subkey = jax.random.split(key)
        offsets = jax.random.normal(subkey, (expansion_factor, num_modes, 4)) * noise_scale # Offset for 4 center coords
        start_idx = i * expansion_factor
        end_idx = start_idx + expansion_factor
        new_normalizations = new_normalizations.at[start_idx:end_idx].set(base_weight/expansion_factor)
        new_centers = base_center + offsets
        new_params = new_params.at[start_idx:end_idx, :, :4].set(new_centers)
        new_params = new_params.at[start_idx:end_idx, :, 4:].set(base_cov)
    total_weight = jnp.sum(new_normalizations)
    new_normalizations = new_normalizations/total_weight
    return new_normalizations, new_params

def plot_wigner(gaussians, filename, exact_state=None):
    x = jnp.linspace(-5, 5, 100)
    p = jnp.linspace(-5, 5, 100)
    X, P = jnp.meshgrid(x, p)
    num_modes = gaussians[1].shape[1]
    
    fig, ax = plt.subplots(num_modes, 3, figsize=(18, 5*num_modes), squeeze=False)
    
    normalizations, params = gaussians
    normalizations = renormalize(normalizations)
    for site in range(num_modes):
        W = jnp.zeros(X.shape)
        for normalization, param in zip(normalizations, params):
            mean, covariance_params = unwrap_params(param)
            mean = mean[site]
            covariance_params = covariance_params[site]
            covariance = calculate_covariance(covariance_params)
            det_cov = jla.det(covariance)
            inv_cov = jla.inv(covariance)
            phase_vars = jnp.array([X.flatten(), P.flatten()]).T
            mean = mean[::2]+ 1j* mean[1::2]
            diff = phase_vars - mean
            exponent = -0.5 * jnp.einsum("ij,ij->i", jnp.dot(diff, inv_cov), diff)
            exponent = exponent.reshape(X.shape)
            W += normalization/(2*jnp.pi*jnp.sqrt(det_cov)) * jnp.real(jnp.exp(exponent))
            
        norm = matplotlib.colors.Normalize(-abs(W).max(), abs(W).max())
        
        if exact_state is not None:
            print(site)
            qt.plot_wigner(exact_state.ptrace(site), xvec=x, yvec=p, ax=ax[site,1], cmap='RdBu_r', colorbar=True)
            w_exact = qt.wigner(exact_state.ptrace(site), xvec=x, yvec=p)
            cf_diff = ax[site,2].contourf(X, P, jnp.abs(w_exact - W), levels=200, cmap='RdBu_r')
            fig.colorbar(cf_diff, ax=ax[site, 2])

        cf = ax[site, 0].contourf(X, P, W, levels=200, cmap='RdBu_r', norm=norm)
        fig.colorbar(cf, ax=ax[site, 0])
        ax[site, 0].set_title(f'Wigner Function Site {site}')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

@jax.jit(static_argnums=(2))
def compute_update_step(t, gaussians, args):
    V_pytree = liouvillian_gradient(gaussians)
    d_norms, d_params = V_pytree
    
    N = d_norms.shape[0]
    num_modes = d_params.shape[1]
    
    V_flat = jnp.concatenate([d_norms, d_params.ravel()])
    
    T = geometric_tensor(gaussians)
    d_combined = jla.solve(T + 1e-12*jnp.eye(T.shape[0]), V_flat)
    
    d_norms_new = d_combined[:N]
    d_params_new = d_combined[N:].reshape(N, num_modes, 6)
    
    return (d_norms_new, d_params_new)

def geometric_tensor_batched(gaussians, batch_size=100):
    """
    Computes Geometric Tensor in batches.
    Correctly handles shapes and vmapping over batches.
    """
    flat_params, unravel_fn = ravel_pytree(gaussians)
    num_params = flat_params.size
    
    # 1. Pad the *Number of Vectors* (Rows), NOT the vector size (Cols)
    remainder = num_params % batch_size
    if remainder != 0:
        pad_rows = batch_size - remainder
        # Create standard (N, N) identity
        basis = jnp.eye(num_params)
        # Pad only the bottom rows with zeros to make total rows divisible by batch_size
        # Result shape: (N + pad, N)
        basis_padded = jnp.pad(basis, ((0, pad_rows), (0, 0)))
    else:
        basis_padded = jnp.eye(num_params)

    # 2. Reshape into batches: (Num_Batches, Batch_Size, Param_Dim)
    num_batches = basis_padded.shape[0] // batch_size
    basis_batches = basis_padded.reshape(num_batches, batch_size, num_params)

    # 3. Define the Core Gradient Function
    def gradient_wrt_ket(bra_params_flat):
        bra_pytree = unravel_fn(bra_params_flat)
        def scalar_overlap(ket_pytree):
            # Recalculate overlap sum
            # Note: Ensure global 'gen_func' and 'num_modes' are accessible or passed in
            return jnp.sum(jax.vmap(lambda g_a: jax.vmap(lambda g_b: gen_func(g_a, g_b, jnp.zeros(4*num_modes)))(bra_pytree))(ket_pytree))
        
        grad_pytree = jax.grad(scalar_overlap)(gaussians)
        return ravel_pytree(grad_pytree)[0]

    # 4. Process Batch Function
    def process_batch(batch_tangents):
        # batch_tangents shape: (Batch_Size, num_params)
        
        # We must VMAP the JVP to handle the batch dimension (128)
        # jvp(func, primals, tangents)
        # primals is fixed (flat_params), tangents varies
        def jvp_wrapper(t):
            _, res = jax.jvp(gradient_wrt_ket, (flat_params,), (t,))
            return res
        
        return jax.vmap(jvp_wrapper)(batch_tangents)

    # 5. Execute Map
    # jax.lax.map executes sequentially to save memory
    T_flat_batched = jax.lax.map(process_batch, basis_batches)
    
    # 6. Reshape and Trim
    # Flatten batches back to matrix rows: (N_padded, N)
    T_flat = T_flat_batched.reshape(-1, num_params)
    
    # Remove the padding rows we added in step 1
    T_flat = T_flat[:num_params, :]
    
    return T_flat

@jax.jit(static_argnums=(2))
def test_compute_update_step(t, gaussians, args):
    # 1. Compute Gradient (V)
    V_pytree = liouvillian_gradient(gaussians)
    V_flat, unravel_fn = ravel_pytree(V_pytree)
    
    # 2. Compute Metric Tensor (T) - Now memory safe
    T = geometric_tensor_batched(gaussians)
    
    # 3. Solve (Add jitter for numerical stability)
    d_combined = jla.solve(T + 1e-12 * jnp.eye(T.shape[0]), V_flat)
    
    # 4. Unflatten
    d_norms, d_params = unravel_fn(d_combined)
    
    return (d_norms, d_params)

def plot_observables(data, num_modes, exact_result=None, filename="observables.png"):
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    for t_eval, gaussians_list in data:
        # gaussians_list is expected to be a list of (norms, params) or just params if norms are fixed/implicit
        # If data comes from sol.ys, it is just params.
        # We need to handle this. For now, assume data is properly formatted or we need to fix the call site.
        # Given the call site in time_evolve returns sol, and user reshapes sol.ys, we need to zip with norms.
        
        # However, to support the tuple structure in vmap, we need a list of tuples.
        # If gaussians_list is a tuple of arrays (norms_t, params_t), vmap works.
        
        for i in range(num_modes):
            n = jax.vmap(partial(number_operator,mode=i))(gaussians_list)
            p = jax.vmap(partial(parity_operator,mode=i))(gaussians_list)
            ax[0].plot(t_eval, n, label=f'VMG n{i}')
            ax[1].plot(t_eval, p, label=f'VMG p{i}')
            print(f"N{i}: {n[-1]}")

    if exact_result is not None:
        # Plot Number operators from expect
        for i in range(num_modes):
            if i < len(exact_result.expect):
                 ax[0].plot(exact_result.times, exact_result.expect[i], '--', label=f'Exact n{i}')
        
        # Calculate and plot Parity operators from states
        if hasattr(exact_result, 'states') and len(exact_result.states) > 0:
            # Infer N from state dimensions
            dims = exact_result.states[0].dims[0]
            N = dims[0]
            
            a_ops = []
            for i in range(num_modes):
                op_list = [qt.qeye(N)] * num_modes
                op_list[i] = qt.destroy(N)
                a_ops.append(qt.tensor(op_list))
            
            n_ops = [a.dag() * a for a in a_ops]
            p_ops = [(1j * np.pi * n).expm() for n in n_ops]
            
            for i in range(num_modes):
                p_ex = qt.expect(p_ops[i], exact_result.states)
                ax[1].plot(exact_result.times, p_ex, '--', label=f'Exact p{i}')

    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('<n>')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('<p>')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def time_evolve(initial_time, end_time, initial_gaussians):
    steps = 300
    t_eval = jnp.linspace(initial_time, end_time, steps)  
    initial_normalization, initial_params = initial_gaussians

    N_G = initial_normalization.shape[0]  
    num_modes = initial_params.shape[1]


    # Log initial observables
    print(f"Starting integration from t={initial_time} to {end_time}...")
    term = ODETerm(test_compute_update_step)
    solver = diffrax.Dopri8()
    saveat = SaveAt(ts=t_eval)

    stepsize_controller = PIDController(rtol=1e-3, atol=1e-6)

    progress_meter = diffrax.TqdmProgressMeter()
    # Flatten params into a pytree-compatible format

    sol = diffeqsolve(term, solver, t0=initial_time, t1=end_time, dt0=None, y0=initial_gaussians, saveat=saveat,
                    stepsize_controller=stepsize_controller, max_steps=None, progress_meter=progress_meter)
        
    print("Integration complete. Computing observables...")
    print(sol.stats)
    
    return sol

def exact_simulation(t, num_sites, t_eval=None):
    N = 12            # Local Hilbert space cutoff (Keep small for many sites!)
    if t_eval is None:
        tlist = np.linspace(0, t, 201)
    else:
        tlist = t_eval

    # --- 2. Construct Operators ---
    # We create a list of annihilation operators for each site in the tensor space
    a_ops = []
    for i in range(num_sites):
        op_list = [qt.qeye(N)] * num_sites
        op_list[i] = qt.destroy(N)
        a_ops.append(qt.tensor(op_list))

    # Derived operators
    n_ops = [a.dag() * a for a in a_ops]
    x_ops = [(a + a.dag()) / np.sqrt(2) for a in a_ops]

    # --- 3. Build Hamiltonian ---
    H = (G / 2) * (a_ops[0].dag()**2) + (np.conj(G) / 2) * (a_ops[0]**2)
    H += (S / 2) * (a_ops[0].dag()) + (np.conj(S) / 2) * (a_ops[0])

    # Local terms: Detuning, Kerr, and Drive
    for i in range(num_sites):
        H += -delta * n_ops[i] 
        H += 0.5 * U * (a_ops[i].dag()**2 * a_ops[i]**2)

    # Interaction terms: Hopping (Nearest Neighbor)
    for i in range(num_sites - 1):
        H += -J * (a_ops[i].dag() * a_ops[i+1] + a_ops[i+1].dag() * a_ops[i])

    # --- 4. Dissipation and Initial State ---
    c_ops = [np.sqrt(gamma) * a for a in a_ops]

    # Initial state: Vacuum on all sites
    psi0 = qt.tensor([qt.basis(N, 0)] * num_sites)

    # --- 5. Parity Operator ---
    # Total parity is the product of local parities
    parity_tot = (1j * np.pi * sum(n_ops)).expm()

    # --- 6. Solve Master Equation ---
    # Define which expectation values to track
    e_ops = n_ops + [parity_tot]

    result = qt.mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options={"store_states":True})
    return result

def plot_centers(gaussians, filename, site=0):
    normalizations, params = gaussians
    # Assuming weights are normalizations.
    weights = renormalize(normalizations)
    x_centers = params[:, site, 0]
    p_centers = params[:, site, 2] # Assuming x, y, p_x, p_y ordering or similar? 
    # unwrap_params: means = params[:, :4]. 
    # mean structure: alpha_a, beta_a. alpha = (q+ip)/sqrt(2)?
    # The code uses mean[::2] and mean[1::2].
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x_centers, p_centers, c=weights, cmap='viridis', alpha=0.8)
    plt.colorbar(sc, label='Weight Magnitude')
    plt.xlabel('X')
    plt.ylabel('P')
    plt.title(f'Gaussian Centers Site {site}')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

@partial(jax.jit, static_argnums=(3,))
def compute_total_wigner(gaussians, X, P, site):
    # params: (N_G, dim)
    # X, P: (H, W)
    
    norms, params = gaussians
    norms = renormalize(norms)
    gaussians = (norms, params)
    
    points = jnp.stack([X.flatten(), P.flatten()], axis=1) # (N_points, 2)
    
    def per_gaussian(gaussian):
        normalization, param = gaussian
        mean, covariance_params = unwrap_params(param)
        
        m_site = mean[site]
        cov_p_site = covariance_params[site]
        
        cov = calculate_covariance(cov_p_site)
        inv_cov = jla.inv(cov)
        det_cov = jla.det(cov)
        
        m_complex = m_site[::2] + 1j * m_site[1::2]
        diff = points - m_complex[None, :] # (N_points, 2)
        
        exponent = -0.5 * jnp.sum(jnp.dot(diff, inv_cov) * diff, axis=1)
        
        return normalization / (2 * jnp.pi * jnp.sqrt(det_cov)) * jnp.real(jnp.exp(exponent))

    terms = jax.vmap(per_gaussian)(gaussians) # (N_G, N_points)
    total = jnp.sum(terms, axis=0) # (N_points,)
    return total.reshape(X.shape)

def animate_wigner(data, filename, t_eval=None, exact_states=None):
    x = jnp.linspace(-5, 5, 100)
    p = jnp.linspace(-5, 5, 100)
    X, P = jnp.meshgrid(x, p)
    
    if isinstance(data, tuple):
        norms, params = data
        num_frames = norms.shape[0]
        num_modes = params.shape[2]
        get_state = lambda i: (norms[i], params[i])
    elif isinstance(data, list):
        num_frames = len(data)
        if num_frames == 0:
            print("Warning: No data provided for Wigner animation.")
            return
        sample_params = data[0][1]
        num_modes = sample_params.shape[1]
        get_state = lambda i: data[i]
    else:
        print("Warning: Unknown data format for Wigner animation.")
        return
    
    print("Precomputing VMG Wigners...")
    vmg_wigners = []
    for site in range(num_modes):
        wigners_list = []
        for t in range(num_frames):
            wigners_list.append(compute_total_wigner(get_state(t), X, P, site))
        vmg_wigners.append(jnp.stack(wigners_list))

    exact_wigners = []
    if exact_states is not None:
        print("Precomputing Exact Wigners...")
        for site in range(num_modes):
            site_wigners = []
            limit = min(len(exact_states), num_frames)
            for t in range(limit):
                rho = exact_states[t].ptrace(site)
                W = qt.wigner(rho, xvec=x, yvec=p)
                site_wigners.append(W)
            exact_wigners.append(np.array(site_wigners))
            
    max_diffs = []
    if exact_states is not None:
        for site in range(num_modes):
            limit = min(len(exact_wigners[site]), len(vmg_wigners[site]))
            diff = jnp.abs(np.array(exact_wigners[site][:limit]) - np.array(vmg_wigners[site][:limit]))
            max_val = float(diff.max())
            max_diffs.append(max_val if max_val > 1e-9 else 1.0)
    
    if exact_states is not None:
        cols = 3
        figsize = (18, 5 * num_modes)
    else:
        cols = 1
        figsize = (6 * num_modes, 5.5)
    
    fig, axes = plt.subplots(num_modes, cols, figsize=figsize, squeeze=False)
    
    if exact_states is not None:
        for site in range(num_modes):
            norm_diff = matplotlib.colors.LogNorm(vmin=1e-8, vmax=max(max_diffs[site], 1e-5))
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_diff)
            sm.set_array([])
            fig.colorbar(sm, ax=axes[site, 2])

    def update(frame):
        
        if t_eval is not None and frame < len(t_eval):
            fig.suptitle(f't = {t_eval[frame]:.3f}', fontsize=14)

        for site in range(num_modes):
            W_vmg = vmg_wigners[site][frame]
            vmax = jnp.abs(W_vmg).max()
            norm = matplotlib.colors.Normalize(-vmax, vmax) if vmax > 1e-9 else None
            
            if exact_states is not None and frame < len(exact_wigners[site]):
                W_exact = exact_wigners[site][frame]
                W_diff = jnp.abs(W_exact - W_vmg)
                
                ax = axes[site, 0]
                ax.clear()
                ax.contourf(X, P, W_vmg, levels=100, cmap='RdBu_r', norm=norm)
                ax.set_title(f'VMG Site {site}')
                
                ax = axes[site, 1]
                ax.clear()
                ax.contourf(X, P, W_exact, levels=100, cmap='RdBu_r', norm=norm)
                ax.set_title(f'Exact Site {site}')
                
                ax = axes[site, 2]
                ax.clear()
                norm_diff = matplotlib.colors.LogNorm(vmin=1e-6, vmax=max(max_diffs[site], 1e-5))
                ax.contourf(X, P, W_diff + 1e-12, levels=100, cmap='RdBu_r', norm=norm_diff)
                ax.set_title(f'Diff Site {site}')
            else:
                ax = axes[site, 0]
                ax.clear()
                ax.contourf(X, P, W_vmg, levels=100, cmap='RdBu_r', norm=norm)
                ax.set_title(f'Site {site}')
                ax.set_xlabel('x')
                ax.set_ylabel('p')
                ax.set_aspect('equal')

        fig.tight_layout(rect=[0, 0, 1, 0.95])

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)
    anim.save(filename)
    plt.close(fig)

# --- L2 Projection Pruning Implementation ---

def compute_overlap(gaussian_a, gaussian_b):
    """
    Computes the overlap integral <W_a | W_b> using the existing gen_func.
    Wigner overlap is equivalent to gen_func with J=0.
    """
    # gaussian_a is (norm, param)
    num_modes = gaussian_a[1].shape[0]
    zeros = jnp.zeros(4 * num_modes)
    
    return gen_func(gaussian_a, gaussian_b, zeros)

@jax.jit
def l2_loss(gaussians_reduced, gaussians_full):
    """
    Calculates L2 distance: || W_reduced - W_full ||^2
    = <W_r|W_r> + <W_f|W_f> - 2<W_r|W_f>
    """
    # Self-interaction of reduced state
    # vmap over pairs
    def sum_overlaps(gs_a, gs_b):
        return jnp.sum(jax.vmap(lambda g_a: jax.vmap(lambda g_b: compute_overlap(g_a, g_b))(gs_b))(gs_a))

    term_rr = sum_overlaps(gaussians_reduced, gaussians_reduced)
    term_ff = sum_overlaps(gaussians_full, gaussians_full)
    term_rf = sum_overlaps(gaussians_reduced, gaussians_full)
    
    return term_ff + term_rr - 2 * term_rf

def repulsion_loss(gaussians, threshold=5e-2):
    norms, params = gaussians
    # 1. Extract Phase Space Centers (Re(alpha), Re(beta))
    # Shape: (N_G, num_modes, 2)
    centers = params[:, :, :4:2] 
    
    centers_flat = centers.reshape(centers.shape[0], -1) 
    # 3. Compute distances between Gaussians in the full multi-mode phase space
    diffs = centers_flat[:, None, :] - centers_flat[None, :, :]
    dist_sq = jnp.sum(diffs**2, axis=-1)
    
    penalty = (1.0 / (dist_sq + 1e-12)) - (1.0 / (threshold**2))
    mask = 1.0 - jnp.eye(dist_sq.shape[0])
    
    return jnp.sum(mask * jax.nn.relu(penalty))

# Update loss function
def total_loss(gaussians_reduced, gaussians_full):
    return l2_loss(gaussians_reduced, gaussians_full) + 0.01 * repulsion_loss(gaussians_reduced)

@jax.jit(static_argnums=(2,3))
def optimize_reduced_state(gaussians_reduced, gaussians_full, steps=200, lr=0.05):
    """
    Performs Adam optimization to adjust the parameters of the reduced Gaussians
    to best fit the full state.
    """
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(gaussians_reduced)
    
    def update(i, carry):
        gaussians, state = carry
        grads = jax.grad(lambda g: total_loss(g, gaussians_full))(gaussians)
        updates, new_state = optimizer.update(grads, state, gaussians)
        new_gaussians = optax.apply_updates(gaussians, updates)
        return new_gaussians, new_state

    final_gaussians, _ = jax.lax.fori_loop(0, steps, update, (gaussians_reduced, opt_state))
    return final_gaussians, total_loss(final_gaussians, gaussians_full)


def l2_prune_params(original_gaussians, initial_gaussians, target_n_gaussians, optimization_steps=500, lr=0.01):
    """
    Reduces the number of Gaussians by:
    1. Sorting by weight magnitude.
    2. Keeping the top `target_n_gaussians`.
    3. Optimizing the remaining Gaussians to minimize L2 error from the original state.
    """
    # original_gaussians is (norms, params)
    orig_norms, orig_params = original_gaussians
    init_norms, init_params = initial_gaussians
    
    # 1. Selection Strategy: Keep largest weights
    weights = jnp.abs(init_norms)
    indices = jnp.argsort(weights)[::-1][:target_n_gaussians]
    
    params_reduced_init = init_params[indices]
    norms_reduced_init = init_norms[indices]
    
    # Renormalize
    norms_reduced_init = norms_reduced_init / jnp.sum(norms_reduced_init)
    
    gaussians_reduced_init = (norms_reduced_init, params_reduced_init)
    
    print(f"Pruning from {init_norms.shape[0]} to {target_n_gaussians} Gaussians via L2 optimization...")
    print(f"Start L2 Loss: {total_loss(gaussians_reduced_init, original_gaussians)}")

    # 2. Optimization Strategy: Minimize L2 distance
    gaussians_optimized, final_loss = optimize_reduced_state(
        gaussians_reduced_init, 
        original_gaussians, 
        steps=optimization_steps, 
        lr=lr
    )
    print(f"Final L2 Loss: {final_loss}")
    
    return gaussians_optimized

end_time = 0.5
num_modes = 30
t_eval = np.linspace(0.0, end_time, 300)
#exact_result = exact_simulation(end_time, num_sites=num_modes, t_eval=t_eval)

gaussians = initialize_vacuum_state(N_G=1, num_modes=num_modes)
gaussians = expand_state_cluster(gaussians, expansion_factor=30, noise_scale=1e-2)

#gaussians = l2_prune_params(gaussians, gaussians, target_n_gaussians=30, optimization_steps=2000, lr=0.02)


sol = time_evolve(0, end_time, gaussians)
final_state = jax.tree.map(lambda x: x[-1], sol.ys)
plot_wigner(final_state, "final_state.png")
plot_observables([(sol.ts, sol.ys)], num_modes, filename="observables.png")
#animate_wigner(sol.ys, "animation.mp4", t_eval=sol.ts, exact_states=exact_result.states)
