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

S = 8 
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

def plot_wigner(params, filename, exact_state=None, num_modes=1):
    x = jnp.linspace(-5, 5, 100)
    p = jnp.linspace(-5, 5, 100)
    X, P = jnp.meshgrid(x, p)
    
    fig, ax = plt.subplots(num_modes, 3, figsize=(18, 5*num_modes), squeeze=False)
    # We can keep the loop here as visualization is not the bottleneck
    # or vectorize it if needed, but loop is fine for plotting at start/end
    params = renormalize_params(params)
    for site in range(num_modes):
        W = jnp.zeros(X.shape)
        for param in params:
            normalization, mean, covariance_params = unwrap_params(param)
            mean = mean[4*site:4*(site+1)]
            covariance_params = covariance_params[2*site:2*(site+1)]
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

def prune_params(params, threshold=1e-2):
    params = renormalize_params(params)
    mask = jnp.abs(params[:, 0]) > threshold
    new_params = params[mask]
    return new_params

@jax.jit(static_argnums=(2))
def compute_update_step(t, flat_params, args):
    N_G = args
    params = flat_params.reshape((N_G,-1))
    params = renormalize_params(params)

    V = liouvillian_gradient(params).flatten()
    T = geometric_tensor(params)


    d_params = jla.solve(T + 1e-12*jnp.eye(T.shape[0]), V)
   #d_params = jla.pinv(T, rcond=1e-12)@V

    return d_params.flatten()

def plot_observables(data, num_modes, exact_result=None, filename="observables.png"):
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    for t_eval, params in data:
        for i in range(num_modes):
            n = jax.vmap(partial(number_operator,mode=i))(params)
            p = jax.vmap(partial(parity_operator,mode=i))(params)
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

    stepsize_controller = PIDController(rtol=1e-3, atol=1e-6)

    progress_meter = diffrax.TqdmProgressMeter()
    # Flatten params into a pytree-compatible format

    sol = diffeqsolve(term, solver, t0=initial_time, t1=end_time, dt0=None, y0=initial_params.flatten(), saveat=saveat,
                    stepsize_controller=stepsize_controller, max_steps=None, progress_meter=progress_meter, args=N_G)
        
    print("Integration complete. Computing observables...")
    print(sol.stats)
    
    # 6. Final Plot and Analysis
    final_params = sol.ys[-1].reshape((N_G, -1))
    final_params = renormalize_params(final_params)

    print("Final Time:", t_eval[-1])
    print("Final Params:\n", final_params)

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

def plot_centers(params, filename, site=0):
    params = renormalize_params(params)
    weights = params[:, 0]
    idx_base = 1 + 4 * site
    x_centers = jnp.real(params[:, idx_base])
    p_centers = jnp.real(params[:, idx_base + 2])

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
def compute_total_wigner(params, X, P, site):
    # params: (N_G, dim)
    # X, P: (H, W)
    
    points = jnp.stack([X.flatten(), P.flatten()], axis=1) # (N_points, 2)
    
    def per_gaussian(p):
        normalization, mean, covariance_params = unwrap_params(p)
        
        m_site = mean[4*site:4*(site+1)]
        cov_p_site = covariance_params[2*site:2*(site+1)]
        
        cov = calculate_covariance(cov_p_site)
        inv_cov = jla.inv(cov)
        det_cov = jla.det(cov)
        
        m_complex = m_site[::2] + 1j * m_site[1::2]
        diff = points - m_complex[None, :] # (N_points, 2)
        
        exponent = -0.5 * jnp.sum(jnp.dot(diff, inv_cov) * diff, axis=1)
        
        return normalization / (2 * jnp.pi * jnp.sqrt(det_cov)) * jnp.real(jnp.exp(exponent))

    terms = jax.vmap(per_gaussian)(params) # (N_G, N_points)
    total = jnp.sum(terms, axis=0) # (N_points,)
    return total.reshape(X.shape)

def animate_wigner(data, filename, t_eval=None, exact_states=None):
    x = jnp.linspace(-5, 5, 100)
    p = jnp.linspace(-5, 5, 100)
    X, P = jnp.meshgrid(x, p)
    
    if len(data) == 0:
        print("Warning: No data provided for Wigner animation.")
        return

    num_modes = (data[0].shape[1] - 1) // 6
    
    print("Precomputing VMG Wigners...")
    vmg_wigners = []
    for site in range(num_modes):
        wigners_list = []
        for t in range(len(data)):
            wigners_list.append(compute_total_wigner(data[t], X, P, site))
        vmg_wigners.append(jnp.stack(wigners_list))

    exact_wigners = []
    if exact_states is not None:
        print("Precomputing Exact Wigners...")
        for site in range(num_modes):
            site_wigners = []
            for t in range(len(exact_states)):
                rho = exact_states[t].ptrace(site)
                W = qt.wigner(rho, xvec=x, yvec=p)
                site_wigners.append(W)
            exact_wigners.append(np.array(site_wigners))
            
    max_diffs = []
    if exact_states is not None:
        for site in range(num_modes):
            diff = jnp.abs(np.array(exact_wigners[site]) - np.array(vmg_wigners[site]))
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
        
        if t_eval is not None:
            fig.suptitle(f't = {t_eval[frame]:.3f}', fontsize=14)

        for site in range(num_modes):
            W_vmg = vmg_wigners[site][frame]
            vmax = jnp.abs(W_vmg).max()
            norm = matplotlib.colors.Normalize(-vmax, vmax) if vmax > 1e-9 else None
            
            if exact_states is not None:
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

    anim = animation.FuncAnimation(fig, update, frames=len(data), interval=50)
    anim.save(filename)
    plt.close(fig)

# --- L2 Projection Pruning Implementation ---

def compute_overlap(params_a, params_b):
    """
    Computes the overlap integral <W_a | W_b> using the existing gen_func.
    Wigner overlap is equivalent to gen_func with J=0.
    """
    num_modes = (params_a.shape[1] - 1) // 6
    zeros = jnp.zeros(4 * num_modes)
    
    # We use the existing gen_func which implements Eq A10 from the paper 
    # calculating the integral of the product of Gaussians.
    # We map over all pairs (i, j)
    pairwise_overlaps = jax.vmap(
        lambda pa: jax.vmap(
            lambda pb: gen_func(pa, pb, zeros)
        )(params_b)
    )(params_a)
    
    return jnp.sum(pairwise_overlaps)

@jax.jit
def l2_loss(params_reduced, params_full):
    """
    Calculates L2 distance: || W_reduced - W_full ||^2
    = <W_r|W_r> + <W_f|W_f> - 2<W_r|W_f>
    """
    # Self-interaction of reduced state
    term_rr = compute_overlap(params_reduced, params_reduced)

    term_ff = compute_overlap(params_full, params_full)
    
    # Cross-interaction (Overlap between Reduced and Full)
    term_rf = compute_overlap(params_reduced, params_full)
    
    # Note: term_ff (full-full overlap) is constant w.r.t gradients, 
    # so we exclude it for optimization speed, but implicitly it completes the square.
    return term_ff + term_rr - 2 * term_rf

def repulsion_loss(params, threshold=5e-2):
    num_modes = (params.shape[1] - 1) // 6
    # Extract centers (real part of means)
    centers = params[:, 1:(1 + 2 * num_modes):2]
    # Compute pairwise distances
    diffs = centers[:, None, :] - centers[None, :, :]
    dist_sq = jnp.sum(diffs**2, axis=-1)
    dist = jnp.sqrt(dist_sq + 1e-12) # 1e-12 prevents NaN gradients at 0
    # Create the "Hinge": value is 0 if dist > threshold
    # violation = max(0, threshold - dist)
    penalty = (1.0 / (dist_sq + 1e-12)) - (1.0 / (threshold**2))
    # Mask diagonal (self-distance is always 0, which triggers violation otherwise)
    mask = 1.0 - jnp.eye(dist.shape[0])
    
    # Sum squared violations
    return jnp.sum(mask * jax.nn.relu(penalty))

# Update loss function
def total_loss(params_reduced, params_full):
    return l2_loss(params_reduced, params_full) + 0.01 * repulsion_loss(params_reduced)

@jax.jit(static_argnums=(2,3))
def optimize_reduced_state(params_reduced, params_full, steps=200, lr=0.05):
    """
    Performs Adam optimization to adjust the parameters of the reduced Gaussians
    to best fit the full state.
    """
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params_reduced)
    
    def update(i, carry):
        params, state = carry
        grads = jax.grad(total_loss)(params, params_full)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state

    final_params, _ = jax.lax.fori_loop(0, steps, update, (params_reduced, opt_state))
    return final_params, total_loss(final_params, params_full)


def l2_prune_params(original_params, initial_params, target_n_gaussians, optimization_steps=500, lr=0.01):
    """
    Reduces the number of Gaussians by:
    1. Sorting by weight magnitude.
    2. Keeping the top `target_n_gaussians`.
    3. Optimizing the remaining Gaussians to minimize L2 error from the original state.
    """
    original_params = renormalize_params(original_params)
    
    # 1. Selection Strategy: Keep largest weights
    weights = jnp.abs(initial_params[:, 0])
    # Get indices of top N weights
    indices = jnp.argsort(weights)[::-1][:target_n_gaussians]
    
    params_reduced_init = initial_params[indices]
    
    # Renormalize the initial guess so it sums to 1 before optimization
    current_weight_sum = jnp.sum(params_reduced_init[:, 0])
    params_reduced_init = params_reduced_init.at[:, 0].divide(current_weight_sum)
    
    print(f"Pruning from {initial_params.shape[0]} to {target_n_gaussians} Gaussians via L2 optimization...")
    print(f"Start L2 Loss: {total_loss(params_reduced_init, original_params)}")

    # 2. Optimization Strategy: Minimize L2 distance
    params_optimized, final_loss = optimize_reduced_state(
        params_reduced_init, 
        original_params, 
        steps=optimization_steps, 
        lr=lr
    )
    print(f"Final L2 Loss: {final_loss}")
    
    # Final renormalization check
    return renormalize_params(params_optimized)

end_time = 0.5
num_modes = 20
t_eval = np.linspace(0.0, end_time, 300)
#exact_result = exact_simulation(0.5, num_sites=2, t_eval=t_eval)


params = initialize_vacuum_state(N_G=1, num_modes=num_modes)
params = expand_state_cluster(params, expansion_factor=30, noise_scale=1e-2)

params = l2_prune_params(params, params, target_n_gaussians=30, optimization_steps=2000, lr=0.02)
plot_wigner(params, "initial_state.png", num_modes=20)

sol = time_evolve(0, end_time, params)
final_params = sol.ys[-1].reshape((params.shape[0], -1))

plot_wigner(final_params, "final_state_20.png", num_modes=num_modes)
plot_observables([(t_eval,sol.ys.reshape((300, -1, 6*2+1)))] , num_modes=num_modes, filename="observables_20.png")

#animate_wigner(sol.ys.reshape((300, -1, 6*2+1)), "test_wigner_movie.mp4", t_eval=t_eval, exact_states=exact_result.states)


'''
sol1 = time_evolve(0, 0.3, params)


final_params = sol1.ys[-1].reshape((params.shape[0], -1))
jnp.save("params.npy", final_params)


final_params = jnp.load("params.npy")

plot_wigner(final_params, "pre_prune_state.png", site=0)
params = expand_state_cluster(final_params, expansion_factor=3, noise_scale=5e-2)
params = l2_prune_params(final_params, params, target_n_gaussians=60, optimization_steps=10000, lr=0.05)
plot_wigner(params, "post_prune_state.png", site=0)
plot_centers(params, "post_prune_centers.png", site=0)
print(params.shape)
sol2 = time_evolve(0.3, 0.5, params)
plot_observables([(np.linspace(0.3, 0.5, 300),sol2.ys.reshape((300, -1, 6*2+1)))] , num_modes=2)

final_params = sol2.ys[-1].reshape((params.shape[0], -1))
plot_wigner(final_params, "final_state.png", exact_state=exact_result.ptrace(0), site=0)

animate_wigner(sol2.ys.reshape((300, -1, 6*2+1)), "wigner_movie.mp4", t_eval=np.linspace(0.3, 0.5, 300))
'''
