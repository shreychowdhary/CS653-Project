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
    orders
):
    """
    Evaluate mixed partial of generating_function using structured orders.

    orders = [(site, phase, multiplicity), ...]
    """
    num_modes = (param_a.shape[0]-1) // 6

    # Zeroth-order case
    if len(orders) == 0:
        return gen_func(
            param_a,
            param_b,
            jnp.zeros(num_modes * 4)
        )

    # Convert to hashable tuple for caching
    orders_tup = tuple(orders)

    flat_orders = structured_orders_to_flat(
        orders_tup,
        num_modes=num_modes    
    )

    g_f = partial(gen_func, param_a, param_b)

    return mixed_partial(
        g_f,
        flat_orders
    )(jnp.zeros(num_modes * 4))

# --- [Physical Constants and Model Definition] ---

S = 1 + 1j
G = 0 + 0j         # Two-photon drive strength
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

def gen_func(params_a, params_b, Js):
    normalization_a, mean_a, covariance_params_a = unwrap_params(params_a)
    normalization_b, mean_b, covariance_params_b = unwrap_params(params_b)
    alpha_a, beta_a = mean_a[::2], mean_a[1::2]
    alpha_b, beta_b =  mean_b[::2], mean_b[1::2]
    J, J_tilde = Js[:Js.size//2], Js[Js.size//2:]
    
    num_modes = covariance_params_a.size // 2
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

def single_photon_drive(param_a, param_b):
    total = 0
    total += jnp.real(S)/jnp.sqrt(2) * gen_func_partial_der(param_a, param_b, [(0,3,1)])
    total -= jnp.imag(S)/jnp.sqrt(2) * gen_func_partial_der(param_a, param_b, [(0,2,1)])
    return total

def double_photon_drive(param_a, param_b):
    total = 0
    for i in range(1):
        total += jnp.real(G) * (
                gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,1)]) +
                gen_func_partial_der(param_a, param_b, [(i,0,1), (i,3,1)])
            )

        total -= jnp.imag(G) * (
            gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,1)]) -
            gen_func_partial_der(param_a, param_b, [(i,1,1), (i,3,1)])
        )
    return total

def delta_term(param_a, param_b):
    total = 0
    num_modes = (param_a.shape[0]-1) // 6

    for i in range(num_modes):
        total += gen_func_partial_der(param_a, param_b, [(i,1,1), (i,2,1)]) - gen_func_partial_der(param_a, param_b, [(i,0,1), (i,3,1)])
    return delta * total

def u_term(param_a, param_b):
    total = 0
    num_modes = (param_a.shape[0]-1) // 6

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

def single_photon_loss_term(param_a, param_b):
    total = 0
    num_modes = (param_a.shape[0]-1) // 6
    for i in range(num_modes):
        total += gen_func_partial_der(param_a, param_b, [(i,0,1), (i,2,1)])
        total += gen_func_partial_der(param_a, param_b, [(i,1,1), (i,3,1)])
        total += 2 * gen_func_partial_der(param_a, param_b, [])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,2,2)])
        total += 0.5 * gen_func_partial_der(param_a, param_b, [(i,3,2)])   
    return gamma / 2 * total

def hopping_term(param_a, param_b):
    total = 0
    num_modes = (param_a.shape[0]-1) // 6
    for i in range(num_modes-1):
        total += gen_func_partial_der(param_a, param_b, [(i,0,1), (i+1,3,1)])
        total += gen_func_partial_der(param_a, param_b, [(i+1,0,1), (i,3,1)])
        total -= gen_func_partial_der(param_a, param_b, [(i,1,1), (i+1,2,1)])
        total -= gen_func_partial_der(param_a, param_b, [(i+1,1,1), (i,2,1)])
    return -J * total

def all_terms(params_a, params_b):
    def func(param_a, param_b):
        return double_photon_drive(param_a, param_b) + delta_term(param_a, param_b) + u_term(param_a, param_b) + single_photon_loss_term(param_a, param_b) + hopping_term(param_a, param_b) + single_photon_drive(param_a, param_b)
    return jnp.sum(jax.vmap(lambda param_a: jax.vmap(lambda param_b: func(param_a, param_b))(params_b))(params_a))


def liouvillian_gradient(params):
    return jax.grad(all_terms, argnums=0) (params, params)

def geometric_tensor(params):
    num_modes = (params.shape[1]-1)//6
    def total_gen_func(params_a, params_b):
        return jnp.sum(jax.vmap(lambda param_a: jax.vmap(lambda param_b: gen_func(param_a, param_b, jnp.zeros(4*num_modes)))(params_a))(params_b))
    return jax.jacfwd(jax.grad(total_gen_func, argnums=(1)), argnums=(0))(params, params).reshape(params.size, params.size)

def renormalize_params(params):
    return params.at[:,0].divide(jnp.sum(params[:,0]))

def number_operator(params, mode):
    params = renormalize_params(params)
    def single_param_n(p):
        normalization, mean, covariance_params = unwrap_params(p)
        covariance = calculate_covariance(covariance_params[2*mode:2*(mode+1)])
        mean = mean[4*mode:4*(mode+1):2]+ 1j* mean[4*mode+1:4*(mode+1):2]
        return normalization*(jnp.sum(jnp.real(mean**2))+covariance[0,0]+covariance[1,1]-1)/2
    
    return jnp.sum(jax.vmap(single_param_n)(params))

def parity_operator(params, mode):
    params = renormalize_params(params)
    def single_param_parity(p):
        normalization, mean, covariance_params = unwrap_params(p)
        covariance = calculate_covariance(covariance_params[2*mode:2*(mode+1)])
        mean = mean[4*mode:4*(mode+1):2]+ 1j* mean[4*mode+1:4*(mode+1):2]
        term = jnp.exp(-0.5 * jnp.dot(-mean, jnp.dot(jla.inv(covariance), -mean)))
        return jnp.pi * normalization * jnp.real(term) / jnp.sqrt((2*jnp.pi)**2 * jnp.linalg.det(covariance))

    return jnp.sum(jax.vmap(single_param_parity)(params))

def unwrap_params(params):
    num_modes = (params.shape[0] - 1) // 6
    normalization = params[0]
    means = params[1:1 + 4 * num_modes]
    covariances = params[1 + 4 * num_modes:]
    return normalization, means, covariances

def initialize_vacuum_state(N_G, num_modes=1):
    params = jnp.zeros((N_G, 1 + 4 * num_modes + 2 * num_modes))
    params = params.at[:,0].set(1.0 / N_G)  #
    return params

def expand_state_cluster(params, expansion_factor=4, noise_scale=1e-4, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    num_modes = (params.shape[1] - 1) // 6
    old_N = params.shape[0]
    new_N = old_N * expansion_factor
    new_params = jnp.zeros((new_N, params.shape[1]))
    print(f"Expanding ansatz from {old_N} to {new_N} Gaussians...")
    for i in range(old_N):
        base_weight = params[i, 0]
        base_center = params[i, 1:1+4*num_modes] # Fixed slicing index from original code
        base_cov = params[i, 1 + 4 * num_modes:]
        key, subkey = jax.random.split(key)
        offsets = jax.random.normal(subkey, (expansion_factor, 4*num_modes)) * noise_scale # Offset for 4 center coords
        start_idx = i * expansion_factor
        end_idx = start_idx + expansion_factor
        new_params = new_params.at[start_idx:end_idx, 0].set(base_weight/expansion_factor)
        new_centers = base_center + offsets
        new_params = new_params.at[start_idx:end_idx, 1:1+4*num_modes].set(new_centers)
        new_params = new_params.at[start_idx:end_idx, 1+4*num_modes:].set(base_cov)
    total_weight = jnp.sum(new_params[:, 0])
    new_params = new_params.at[:, 0].divide(total_weight)
    return new_params

def plot_wigner(params, filename):
    x = jnp.linspace(-5, 5, 100)
    p = jnp.linspace(-5, 5, 100)
    X, P = jnp.meshgrid(x, p)
    W = jnp.zeros(X.shape)
    
    # We can keep the loop here as visualization is not the bottleneck
    # or vectorize it if needed, but loop is fine for plotting at start/end
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
    
    plt.title('Wigner Function')
    plt.savefig(filename)

@jax.jit(static_argnums=(2))
def compute_update_step(t, flat_params, args):
    N_G = args
    params = flat_params.reshape((N_G,-1))
    params = renormalize_params(params)

    V = liouvillian_gradient(params).flatten()
    T = geometric_tensor(params)


    d_params = jla.solve(T + 1e-12*jnp.eye(T.shape[0]), V)

    return d_params.flatten()

def time_evolve(initial_time, end_time, initial_params):
    steps = 300
    t_eval = jnp.linspace(initial_time, end_time, steps)  
    N_G = initial_params.shape[0]  
    num_modes = (initial_params.shape[1] - 1) // 6


    # Log initial observables
    print(f"Starting integration from t={initial_time} to {end_time}...")
    term = ODETerm(compute_update_step)
    solver = diffrax.Dopri8()
    saveat = SaveAt(ts=t_eval)

    stepsize_controller = PIDController(rtol=1e-4, atol=1e-7)

    progress_meter = diffrax.TqdmProgressMeter()
    # Flatten params into a pytree-compatible format

    sol = diffeqsolve(term, solver, t0=initial_time, t1=end_time, dt0=None, y0=initial_params.flatten(), saveat=saveat,
                    stepsize_controller=stepsize_controller, max_steps=None, progress_meter=progress_meter, args=N_G)
        
    print("Integration complete. Computing observables...")
    print(sol.stats)
    
    # 6. Final Plot and Analysis
    final_params = sol.ys[-1].reshape((N_G, -1))

    print("Final Time:", t_eval[-1])
    print("Final Params:\n", final_params)

    fig, ax = plt.subplots(1, 2, figsize=(18, 5))

    for i in range(num_modes):
        n = jax.vmap(partial(number_operator,mode=i))(sol.ys.reshape((steps, N_G, -1)))
        p = jax.vmap(partial(parity_operator,mode=i))(sol.ys.reshape((steps, N_G, -1)))
        ax[0].plot(t_eval, n, label=f'n{i}')
        print(f"N{i}: {n[-1]}")

    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('<n>')
    ax[0].legend()
    ax[0].grid(True)

    plt.tight_layout()
    plt.savefig("observables.png")
    return sol

params = initialize_vacuum_state(N_G=1, num_modes=3)
params = expand_state_cluster(params, expansion_factor=40, noise_scale=1e-2)
sol = time_evolve(0, 10, params)
