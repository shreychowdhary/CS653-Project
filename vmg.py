import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax
import jax.scipy.linalg as jsl
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
from typing import Union, Tuple, Iterator
import fractions
import itertools
import functools
import math
import numpy as np
from diffrax import diffeqsolve, Dopri8, Tsit5, ODETerm, SaveAt, PIDController
import diffrax
import optimistix as optx

import jax
import jax.numpy as jnp
from jax.experimental import jet
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
    num_modes: int,
    n_phase: int
) -> MultiIndex:
    """
    Convert [(site, phase, multiplicity), ...] to flat multi-index.
    """
    D = num_modes * n_phase
    flat = [0] * D

    for site, phase, mult in orders:
        if mult == 0:
            continue
        if site < 0 or site >= num_modes:
            raise ValueError(f"Invalid site index {site}")
        if phase < 0 or phase >= n_phase:
            raise ValueError(f"Invalid phase index {phase}")
        if mult < 0:
            raise ValueError("Multiplicity must be non-negative")

        flat[site * n_phase + phase] += mult

    return tuple(flat)


def mixed_partial(f, orders: MultiIndex):
    """
    Compute mixed partial derivative specified by flat multi-index `orders`
    using JAX jets. Works for arbitrary dimension.
    """
    d = sum(orders)
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

def gen_func_partial_der(
    param_a,
    param_b,
    orders,
    num_modes: int = 1,
    n_phase: int = 4,
):
    """
    Evaluate mixed partial of generating_function using structured orders.

    orders = [(site, phase, multiplicity), ...]
    """

    # Zeroth-order case
    if len(orders) == 0:
        return gen_func(
            param_a,
            param_b,
            jnp.zeros(num_modes * n_phase)
        )

    # Convert to hashable tuple for caching
    orders_tup = tuple(orders)

    flat_orders = structured_orders_to_flat(
        orders_tup,
        num_modes=num_modes,
        n_phase=n_phase
    )

    g_f = partial(gen_func, param_a, param_b)

    return mixed_partial(
        g_f,
        flat_orders
    )(jnp.zeros(num_modes * n_phase))

# --- [Physical Constants and Model Definition] ---

G = 30 + 30j
delta = 0
U = 10
gamma = 0
eta = 10
J = 2

def calculate_covariance(covariance_params):
    r,phi = covariance_params
    S = jnp.array([
        [jnp.cos(phi), jnp.sin(phi)],
        [jnp.sin(phi), -jnp.cos(phi)]
    ])
    Sigma = 0.5*jnp.cosh(2*r)*jnp.identity(2)-0.5*jnp.sinh(2*r)*S
    return Sigma

def covariance_sum_inv(covariance_params_a, covariance_params_b):
    ra, phia = covariance_params_a
    rb, phib = covariance_params_b
    
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

def unwrap_params(params):
    normalization = params[0]
    mean = params[1:5]
    covariance_params = params[5:7]
    return normalization, mean, covariance_params

def gen_func(params_a, params_b, Js):
    normalization_a, mean_a, covariance_params_a = unwrap_params(params_a)
    normalization_b, mean_b, covariance_params_b = unwrap_params(params_b)
    alpha_a, beta_a = mean_a[::2], mean_a[1::2]
    alpha_b, beta_b =  mean_b[::2], mean_b[1::2]
    J, J_tilde = Js[:Js.size//2], Js[Js.size//2:]
    
    M_modes = 1
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
    log_denominator = 0.5 * (2*M_modes * jnp.log(2 * jnp.pi) + logdet)
    prefactor = 0.5 * jnp.exp(num_exponent - log_denominator)
    
    return normalization_a*normalization_b*prefactor * Z1 * (M_minus * C_minus + M_plus * C_plus)

@jax.jit
def geometric_tensor(params):
    def total_gen_func(params_a, params_b):
        params_a = duplicate_with_neg_mean(params_a)
        params_b = duplicate_with_neg_mean(params_b)
        return jnp.sum(jax.vmap(lambda param_a: jax.vmap(lambda param_b: gen_func(param_a, param_b, jnp.zeros((4))))(params_a))(params_b))
    return jax.jacfwd(jax.grad(total_gen_func, argnums=(1)), argnums=(0))(params, params).reshape(params.size, params.size)

def g_term(param_a, param_b, num_modes=1):
    total = 0
    for i in range(num_modes):
        total += jnp.real(G) * (
                gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,1)]) +
                gen_func_partial_der(param_a, param_b, [(i,0,1), (i,3,1)])
            )

        total -= jnp.imag(G) * (
            gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,1)]) -
            gen_func_partial_der(param_a, param_b, [(i,1,1), (i,3,1)])
        )
    return total


def delta_term(param_a, param_b, num_modes=1):
    total = 0
    for i in range(num_modes):
        total += gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,1)]) - gen_func_partial_der(param_a, param_b, [(i,0,1), (i,3,1)])
    return delta * total

def u_term(param_a, param_b, num_modes=1):
    total = 0
    for i in range(num_modes):
        total += gen_func_partial_der(param_a, param_b, [(i,0,3), (i,3,1)])
        total += gen_func_partial_der(param_a, param_b, [(i,0,1), (i,1,2), (i,3,1)])
        total += -gen_func_partial_der(param_a, param_b, [(i,0,2), (i,1,1), (i,2,1)])
        total += -gen_func_partial_der(param_a, param_b, [(i,1,3), (i,2,1)])
        total += 2 * gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,1)])
        total += -2 * gen_func_partial_der(param_a, param_b, [(i,0,1), (i,3,1)])
        total += 0.25 * gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,3)])
        total += -0.25 * gen_func_partial_der(param_a, param_b, [(i,0,1), (i,3,3)])
        total += -0.25 * gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,2), (i,3,1)])
        total += 0.25 * gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,1), (i,3,2)])
    return U / 2 * total

def single_photon_loss_term(param_a, param_b, num_modes=1):
    total = 0
    for i in range(num_modes):
        total += gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,1)])
        total += gen_func_partial_der(param_a, param_b, [(i,1,1), (i,3,1)])
        total += 2 * gen_func_partial_der(param_a, param_b, [])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,2,2)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,3,2)])   
    return gamma / 2 * total

def double_photon_loss_term(param_a, param_b, num_modes=1):
    total = 0
    for i in range(num_modes):
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,0,3), (i,2,1)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,0,1), (i,1,2), (i,2,1)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,1,3), (i,3,1)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,0,2), (i,1,1), (i,3,1)])
        total += 2 * gen_func_partial_der(param_a, param_b, [(i,0,2)])
        total += 2 * gen_func_partial_der(param_a, param_b, [(i,1,2)])
        total += gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,1)])
        total += gen_func_partial_der(param_a, param_b, [(i,1,1), (i,3,1)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,0,2), (i,2,2)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,0,2), (i,3,2)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,1,2), (i,2,2)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,1,2), (i,3,2)])
        total += 0.125 * gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,3)])
        total += 0.125 * gen_func_partial_der(param_a, param_b, [(i,1,1), (i,3,3)])
        total += 0.125 * gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,1), (i,3,2)])
        total += 0.125 * gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,2), (i,3,1)])
    return eta * total

def hopping_term(param_a, param_b, num_modes=1):
    total = 0
    for i in range(num_modes-1):
        total += gen_func_partial_der(param_a, param_b, [(i,0,1), (i+1,3,1)])
        total += gen_func_partial_der(param_a, param_b, [(i+1,0,1), (i,3,1)])
        total -= gen_func_partial_der(param_a, param_b, [(i,1,1), (i+1,2,1)])
        total -= gen_func_partial_der(param_a, param_b, [(i+1,1,1), (i,2,1)])

    return -J * total


def all_terms(params_a, params_b):
    params_a = duplicate_with_neg_mean(params_a)
    params_b = duplicate_with_neg_mean(params_b)
    def func(param_a, param_b):
        return g_term(param_a, param_b) + delta_term(param_a, param_b) + u_term(param_a, param_b) + single_photon_loss_term(param_a, param_b) + double_photon_loss_term(param_a, param_b)
    return jnp.sum(jax.vmap(lambda param_a: jax.vmap(lambda param_b: func(param_a, param_b))(params_b))(params_a))

@jax.jit
def liouvillian_gradient(params):
    return jax.grad(all_terms, argnums=0) (params, params)

def renormalize_params(params):
    total_weight = jnp.sum(params[:, 0])
    return params.at[:, 0].set(params[:, 0]/ total_weight)

@jax.jit
def number_operator(params):
    params = duplicate_with_neg_mean(params)
    params = renormalize_params(params)
    # Vectorized helper for number operator within vmap
    def single_param_n(p):
        normalization, mean, covariance_params = unwrap_params(p)
        covariance = calculate_covariance(covariance_params)
        mean = mean[::2]+ 1j* mean[1::2]
        return normalization*(jnp.sum(jnp.real(mean**2))+covariance[0,0]+covariance[1,1]-1)/2
    
    return jnp.sum(jax.vmap(single_param_n)(params))

@jax.jit
def parity_operator(params):
    # Vectorized helper for parity operator within vmap
    params = duplicate_with_neg_mean(params)
    params = renormalize_params(params)
    def single_param_parity(p):
        normalization, mean, covariance_params = unwrap_params(p)
        covariance = calculate_covariance(covariance_params)
        mean = mean[::2]+ 1j* mean[1::2] # Used in original, but math below uses real vector mean
        term = jnp.exp(-0.5 * jnp.dot(-mean, jnp.dot(jla.inv(covariance), -mean)))
        return jnp.pi * normalization * jnp.real(term) / jnp.sqrt((2*jnp.pi)**2 * jnp.linalg.det(covariance))

    return jnp.sum(jax.vmap(single_param_parity)(params))

def plot_wigner(params, filename, exact_state=None):
    x = jnp.linspace(-5, 5, 100)
    p = jnp.linspace(-5, 5, 100)
    X, P = jnp.meshgrid(x, p)
    W = jnp.zeros(X.shape)
    
    # We can keep the loop here as visualization is not the bottleneck
    # or vectorize it if needed, but loop is fine for plotting at start/end
    params = duplicate_with_neg_mean(params)
    params = renormalize_params(params)
    for param in params:
        normalization, mean, covariance_params = unwrap_params(param)
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
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    cf = ax[0].contourf(X, P, W, levels=200, cmap='RdBu_r', norm=norm)
    fig.colorbar(cf, ax=ax[0])
    if exact_state is not None:
        qt.plot_wigner(exact_state, xvec=x, yvec=p, ax=ax[1], cmap='RdBu_r', colorbar=True)
        w_exact = qt.wigner(exact_state, xvec=x, yvec=p)
        cf_diff = ax[2].contourf(X, P, jnp.abs(w_exact - W), levels=200, cmap='RdBu_r')
        fig.colorbar(cf_diff, ax=ax[2])
    
    plt.title('Wigner Function')
    #plt.savefig(filename)


def initialize_vacuum_state(N_G):
    params = jnp.zeros((N_G, 7))
    params = params.at[:, 0].set(1.0)
    params = params.at[:,0].divide(jnp.sum(params[:,0]))
    return params

def expand_state_cluster(params, expansion_factor=4, noise_scale=1e-4, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    old_N = params.shape[0]
    new_N = old_N * expansion_factor
    new_params = jnp.zeros((new_N, 7))
    print(f"Expanding ansatz from {old_N} to {new_N} Gaussians...")
    for i in range(old_N):
        base_weight = params[i, 0]
        base_center = params[i, 1:5] # Fixed slicing index from original code
        base_cov = params[i, 5:7]
        key, subkey = jax.random.split(key)
        offsets = jax.random.normal(subkey, (expansion_factor, 4)) * noise_scale # Offset for 4 center coords
        start_idx = i * expansion_factor
        end_idx = start_idx + expansion_factor
        random_weight = jax.random.uniform(subkey, (expansion_factor,)) 
        new_params = new_params.at[start_idx:end_idx, 0].set(random_weight)
        new_centers = base_center + offsets
        new_params = new_params.at[start_idx:end_idx, 1:5].set(new_centers)
        new_params = new_params.at[start_idx:end_idx, 5:7].set(base_cov)
    total_weight = jnp.sum(new_params[:, 0])
    new_params = new_params.at[:, 0].divide(total_weight)
    return new_params

def duplicate_with_neg_mean(params):
    neg_params = params.at[:, 1:5].set(-params[:, 1:5])
    params = jnp.concatenate([params, neg_params], axis=0)
    params = params.at[:, 0].divide(2)
    return params

@jax.jit
def compute_update_step(t, params_flat, args):
    params = params_flat.reshape((-1,7))
    params = renormalize_params(params)
    # Calculate Gradient and Geometric Tensor
    V = liouvillian_gradient(params).flatten()
    T = geometric_tensor(params)
    # Logging for diagnostics
    def log_fn(args):
        t_val = args
        jax.debug.print(
            "t: {t_v} | N: {m} | P: {p}",
            t_v=t_val, m=number_operator(params), p=parity_operator(params)
        )

    d_params = jla.solve(T + 1e-12*jnp.eye(T.shape[0]), V)
    #d_params = jla.lstsq(T, V, rcond=1e-12)[0]
    log_condition = jnp.logical_and(jnp.mod(t, 0.001) < 1e-6, t > 1e-6)
    jax.lax.cond(
        log_condition,
        log_fn,
        lambda _: None,
        t,
    )

    return d_params.flatten()

def time_evolve(initial_time, end_time, params=None, N_G=6):
    # Simulation parameters
    N = 100            # Hilbert space cutoff (must be high enough to capture squeezing)
    steps = 100       # Higher resolution than the Euler step
    t_eval = jnp.linspace(initial_time, end_time, steps)

    # --- 2. Operators ---
    a = qt.destroy(N)
    n_op = a.dag() * a
    x_op = (a + a.dag()) / np.sqrt(2) # Position quadrature
    p_op = 1j * (a.dag() - a) / np.sqrt(2) # Momentum quadrature

    # Parity operator defined as exp(i * pi * n)
    # Equivalently, it is a diagonal matrix with entries (-1)^n
    parity_op = (1j * np.pi * n_op).expm() 

    # --- 3. Hamiltonian Construction ---
    # H_detuning = delta * a^dag * a
    H_det = -delta * n_op

    # H_kerr = (U/2) * a^dag^2 * a^2 (Standard Kerr form)
    H_kerr = 0.5 * U * a.dag()**2 * a**2

    # H_drive = G * a^dag^2 + G* * a^2 (Two-photon drive)
    # Note: In some conventions, this might be i(G a^dag^2 - G* a^2).
    # Based on your JAX summation, it implies a direct Hermitian addition:
    H_drive = (G/2 * a.dag()**2) + (np.conj(G)/2 * a**2)

    # Total Hamiltonian
    # To simulate the "Only G term" case, set delta=0 and U=0 here.
    H = H_drive + H_det + H_kerr

    # --- 4. Dissipation ---
    # Single photon loss: sqrt(gamma) * a
    c_ops = [np.sqrt(gamma) * a, np.sqrt(eta) * a**2]

    # --- 5. Initial State ---
    # Vacuum state
    psi0 = (qt.coherent(N, 2) + qt.coherent(N, -2)).unit()
    psi0 = qt.basis(N, 0)

    # --- 6. Solve Master Equation ---
    result = qt.mesolve(H, psi0, t_eval, c_ops, e_ops=[x_op, p_op, n_op, parity_op], options={"store_final_state":True})

    # --- 7. Plotting & Verification ---
    means_x = result.expect[0]
    means_p = result.expect[1]
    means_n = result.expect[2]
    means_parity = result.expect[3]
    #params = initialize_cat_state(alpha=2)
    if params is None:
        params = initialize_vacuum_state(N_G=1)
        params = expand_state_cluster(params, expansion_factor=N_G, noise_scale=1e-2)
    param_shape = params.shape
    
    # Initial Plot
    print("Plotting initial state...")
    plot_wigner(params, "initial_state.png", exact_state=psi0)

    
    print(f"Starting integration from t={initial_time} to {end_time}...")
    term = ODETerm(compute_update_step)
    solver = diffrax.Dopri8()
    saveat = SaveAt(ts=t_eval)

    stepsize_controller = PIDController(rtol=1e-3, atol=1e-6)

    progress_meter = diffrax.TextProgressMeter()
    sol = diffeqsolve(term, solver, t0=initial_time, t1=end_time, dt0=None, y0=params.flatten(), saveat=saveat,
                    stepsize_controller=stepsize_controller, max_steps=None, progress_meter=progress_meter)
        
    print("Integration complete. Computing observables...")
    print(sol.stats)

    # 5. Post-process Observables using JAX Vmap (Much faster than looping)
    # sol.y shape is (num_params, num_timesteps). Transpose to (num_timesteps, num_params)
    time_evolution_flat = sol.ys
    # Reshape to (num_timesteps, N_G, 8)
    time_evolution_params = time_evolution_flat.reshape(-1, *param_shape)
    
    # Vectorize observable functions over the time axis
    compute_all_n = jax.vmap(number_operator)
    compute_all_parity = jax.vmap(parity_operator)
    
    ns = compute_all_n(time_evolution_params)
    parities = compute_all_parity(time_evolution_params)

    print("Initial:", ns[0], parities[0])
    
    # 6. Final Plot and Analysis
    final_params = time_evolution_params[-1]
    #final_params = renormalize_params(final_params)

    plot_wigner(final_params, "final_state.png", exact_state=result.final_state)
    print(result)
    print("Final Time:", t_eval[-1])
    print("Final Params:\n", final_params)
    print("Final Number Operator:", ns[-1])
    print("Final Parity Operator:", parities[-1])

    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    # Plot Observables
    ax[0].plot(t_eval, ns, label='Number Operator')
    ax[0].plot(t_eval, means_n, '--') 
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('<n>')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(t_eval, parities, label='Parity')
    ax[1].plot(t_eval, means_parity, '--')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Parity')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("observables.png")
    return sol

sol = time_evolve(0, 0.05, N_G=16)

# params should be list of dictionaries
# weight, means, covariances
# update expand func
# 