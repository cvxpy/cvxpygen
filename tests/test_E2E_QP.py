
import pytest
import cvxpy as cp
import numpy as np
import glob
import os
import io
import importlib
import itertools
import utils_test
import sys
sys.path.append('../')
from cvxpygen import cpg


def actuator_problem():

    # define dimensions (test for degenerate vectors and matrices)
    n, m = 1, 3

    # define variables
    u = cp.Variable(n, name='u')
    delta_u = cp.Variable((n, n), name='delta_u')

    # define parameters
    A = cp.Parameter((m, n), name='A')
    w = cp.Parameter(m, name='w')
    lamb_sm = cp.Parameter(nonneg=True, name='lamb_sm')
    kappa = cp.Parameter(n, nonneg=True, name='kappa')
    u_prev = cp.Parameter(n, name='u_prev')
    u_min = cp.Parameter(n, name='u_min')
    u_max = cp.Parameter(n, name='u_max')

    # define objective
    objective = cp.Minimize(cp.sum_squares(A @ u - w) + lamb_sm * cp.sum_squares(delta_u) + kappa @ cp.abs(u))

    # define constraints
    constraints = [u_min <= u, u <= u_max, delta_u == u - u_prev]

    # define problem
    return cp.Problem(objective, constraints)


def MPC_problem():

    # define dimensions
    H, n, m = 10, 6, 3

    # define variables
    U = cp.Variable((m, H), name='U')
    X = cp.Variable((n, H + 1), name='X')

    # define parameters
    Psqrt = cp.Parameter((n, n), name='Psqrt', diag=True)
    Qsqrt = cp.Parameter((n, n), name='Qsqrt', diag=True)
    Rsqrt = cp.Parameter((m, m), name='Rsqrt', diag=True)
    nonzeros_A = [(i, i) for i in range(n)] + [(i, 3+i) for i in range(n // 2)]
    A = cp.Parameter((n, n), name='A', sparsity=tuple(zip(*nonzeros_A)))
    nonzeros_B = [(3+i, i) for i in range(n // 2)]
    B = cp.Parameter((n, m), name='B', sparsity=tuple(zip(*nonzeros_B)))
    x_init = cp.Parameter(n, name='x_init')

    # define objective
    objective = cp.Minimize(
        cp.sum_squares(Psqrt @ X[:, H - 1]) + cp.sum_squares(Qsqrt @ X[:, :H]) + cp.sum_squares(Rsqrt @ U)+1)

    # define constraints
    constraints = [X[:, 1:] == A @ X[:, :H] + B @ U,
                   cp.abs(U) <= 1,
                   X[:, 0] == x_init]

    # define problem
    return cp.Problem(objective, constraints)


def portfolio_problem():

    # define dimensions
    n, m = 50, 10

    # define variables
    w = cp.Variable(n, name='w')
    delta_w = cp.Variable(n, name='delta_w')
    f = cp.Variable(m, name='f')

    # define parameters
    a = cp.Parameter(n, name='a')
    F = cp.Parameter((n, m), name='F')
    Sig_f_sqrt = cp.Parameter((m, m), name='Sig_f_sqrt')
    d_sqrt = cp.Parameter(n, name='d_sqrt')
    k_tc = cp.Parameter(n, nonneg=True, name='k_tc')
    k_sh = cp.Parameter(n, nonneg=True, name='k_sh')
    w_prev = cp.Parameter(n, name='w_prev')
    L = cp.Parameter(nonneg=True, name='L')

    # define objective
    objective = cp.Maximize(a @ w
                            - cp.sum_squares(Sig_f_sqrt @ f)
                            - cp.sum_squares(cp.multiply(d_sqrt, w))
                            - k_tc @ cp.abs(delta_w)
                            + k_sh @ cp.minimum(0, w))

    # define constraints
    constraints = [f == F.T @ w,
                   np.ones(n) @ w == 1,
                   cp.norm(w, 1) <= L,
                   delta_w == w - w_prev]

    # define problem
    return cp.Problem(objective, constraints)


def assign_data(prob, name, seed):

    np.random.seed(seed)

    if name == 'actuator':

        prob.param_dict['A'].value = np.array([[1], [1], [1]])
        prob.param_dict['w'].value = np.array([2, 3, 5])
        prob.param_dict['lamb_sm'].value = np.random.rand()
        prob.param_dict['kappa'].value = 0.1 * np.ones(1)
        prob.param_dict['u_prev'].value = 0 * np.ones(1)
        prob.param_dict['u_min'].value = -1 * np.ones(1)
        prob.param_dict['u_max'].value = 1 * np.ones(1)

    elif name == 'MPC':

        # continuous-time dynmaics
        A_cont = np.concatenate((np.array([[0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]]),
                                           np.zeros((3, 6))), axis=0)
        mass = 1
        B_cont = np.concatenate((np.zeros((3, 3)),
                                 (1 / mass) * np.diag(np.ones(3))), axis=0)

        # discrete-time dynamics
        td = 0.1

        prob.param_dict['A'].value = np.eye(6) + td * A_cont
        prob.param_dict['B'].value = td * B_cont
        prob.param_dict['Psqrt'].value = np.eye(6)
        prob.param_dict['Qsqrt'].value = np.eye(6)
        prob.param_dict['Rsqrt'].value = np.sqrt(0.1) * np.eye(3)
        prob.param_dict['x_init'].value = -2*np.ones(6) + 4*np.random.rand(6)

    elif name == 'portfolio':

        n, m = 50, 10
        gamma = 1
        alpha = np.random.randn(n)
        kappa_tc = 0.01 * np.ones(n)
        kappa_sh = 0.05 * np.ones(n)
        prob.param_dict['a'].value = alpha / gamma
        prob.param_dict['F'].value = np.round(np.random.randn(n, m))
        prob.param_dict['Sig_f_sqrt'].value = np.diag(np.random.rand(m))
        prob.param_dict['d_sqrt'].value = np.random.rand(n)
        prob.param_dict['k_tc'].value = kappa_tc / gamma
        prob.param_dict['k_sh'].value = kappa_sh / gamma
        prob.param_dict['w_prev'].value = np.zeros(n)
        prob.param_dict['L'].value = 1.6

    return prob


def get_primal_vec(prob, name):
    if name == 'actuator':
        return np.array((prob.var_dict['u'].value.squeeze(), prob.var_dict['delta_u'].value.squeeze()))
    elif name == 'MPC':
        return np.concatenate((prob.var_dict['U'].value.flatten(), prob.var_dict['X'].value.flatten()))
    elif name == 'portfolio':
        return np.concatenate((prob.var_dict['w'].value, prob.var_dict['delta_w'].value, prob.var_dict['f'].value))


N_RAND = 2

name_solver_seed = [['actuator', 'MPC', 'portfolio'],
                    ['OSQP', 'SCS', 'QOCO', 'QOCOGEN'],
                    list(np.arange(N_RAND))]

name_to_prob = {'actuator': actuator_problem(),
                'MPC': MPC_problem(),
                'portfolio': portfolio_problem()}


@pytest.mark.parametrize('name, solver, seed', list(itertools.product(*name_solver_seed)))
def test(name, solver, seed):

    prob = name_to_prob[name]

    if seed == 0:
        prob = assign_data(prob, name, 0)
        identifier = f'test_{name}_{solver}'
        if solver == 'OSQP':
            cpg.generate_code(prob, code_dir=identifier, prefix=identifier)  # test automatic solver selection
        else:
            cpg.generate_code(prob, code_dir=identifier, prefix=identifier, solver=solver)
        assert len(glob.glob(os.path.join(identifier, 'cpg_module.*'))) > 0

    prob = assign_data(prob, name, seed)

    val_py, prim_py, dual_py, val_cg, prim_cg, dual_cg, prim_py_norm, dual_py_norm, stats_py, stats_cg, sol_cg = \
        utils_test.check(prob, solver, name, get_primal_vec)

    if not np.isinf(val_py):
        assert abs((val_cg - val_py) / val_py) < 0.1

    if prim_py_norm > 1e-6:
        assert np.linalg.norm(prim_cg - prim_py, 2) / prim_py_norm < 0.1
    else:
        assert np.linalg.norm(prim_cg, 2) < 1e-3

    if dual_py_norm > 1e-6:
        assert np.linalg.norm(dual_cg - dual_py, 2) / dual_py_norm < 0.1
    else:
        assert np.linalg.norm(dual_cg, 2) < 1e-3
        
    # QOCOGEN is not in CVXPY, but QOCO is an identical (but non-customized) solver, so to check QOCOGEN, we use QOCO.
    if solver == 'QOCOGEN':
        assert stats_cg.solver_name == 'QOCOGEN'
    else:
        assert stats_py.solver_name == stats_cg.solver_name
    assert sol_cg.opt_val == val_cg

    
def test_OSQP_verbose():

    prob = actuator_problem()
    prob = assign_data(prob, 'actuator', 0)

    cpg.generate_code(prob, code_dir='test_actuator_OSQP_verbose', solver='OSQP', prefix='actuator_OSQP_verbose', enable_settings=['verbose'])
    assert len(glob.glob(os.path.join('test_actuator_OSQP_verbose', 'cpg_module.*'))) > 0

    module = importlib.import_module('test_actuator_OSQP_verbose.cpg_solver')
    prob.register_solve('CPG', module.cpg_solve)

    prob = assign_data(prob, 'actuator', 0)

    verbose_output = io.StringIO()
    sys.stdout = verbose_output
    
    _ = utils_test.check(prob, 'OSQP', 'actuator', get_primal_vec, verbose=False)
    assert 'optimal objective' not in verbose_output.getvalue()

    _ = utils_test.check(prob, 'OSQP', 'actuator', get_primal_vec, verbose=True)
    assert 'optimal objective' in verbose_output.getvalue()

    sys.stdout = sys.__stdout__


def test_clarabel():
    test('actuator', 'CLARABEL', 0)
    test('actuator', 'CLARABEL', 1)


def dense_P_problem():

    # A dense quadratic objective: the solver gets a dense P. OSQP, SCS,
    # Clarabel and QOCO all expect only its upper triangle.
    n = 4
    W = np.random.default_rng(0).standard_normal((n, n))
    x = cp.Variable(n, name='x')
    q = cp.Parameter(n, name='q')
    c = cp.Parameter(nonneg=True, name='c')
    objective = cp.Minimize(cp.quad_form(x, W @ W.T + np.eye(n)) + q @ x)
    constraints = [cp.sum(x) == c, x >= -1]
    return cp.Problem(objective, constraints)


def solve_dense_P(solver, identifier, n_solves):
    """Generate code for dense_P_problem and compare n_solves solves with CVXPY."""
    prob = dense_P_problem()
    rng = np.random.default_rng(1)
    prob.param_dict['q'].value = rng.standard_normal(4)
    prob.param_dict['c'].value = 1.0

    cpg.generate_code(prob, code_dir=identifier, prefix=identifier, solver=solver)
    module = importlib.import_module(f'{identifier}.cpg_solver')
    prob.register_solve('CPG', module.cpg_solve)

    x = prob.var_dict['x']
    settings = {'eps_abs': 1e-9, 'eps_rel': 1e-9} if solver == 'OSQP' else {}
    for _ in range(n_solves):
        val_ref = prob.solve(solver='CLARABEL')
        x_ref = x.value
        val_cg = prob.solve(method='CPG', **settings)
        assert np.allclose(x.value, x_ref, atol=1e-4)
        assert np.isclose(val_cg, val_ref, rtol=1e-4)
        prob.param_dict['q'].value = rng.standard_normal(4)
        prob.param_dict['c'].value = 2.0


@pytest.mark.parametrize('solver', [
    'OSQP', 'SCS', 'CLARABEL',
    pytest.param('QOCO', marks=pytest.mark.xfail(
        strict=True, reason='Re-solving with generated QOCO code ends in a numerical '
                            'error, also for diagonal P; only the first solve is correct.')),
])
def test_dense_P(solver):
    # The first solve fails if P is not upper triangular; later solves re-use
    # the generated solver with new parameter values.
    solve_dense_P(solver, f'test_dense_P_{solver}', n_solves=3)


def test_dense_P_QOCO_first_solve():
    solve_dense_P('QOCO', 'test_dense_P_QOCO_first', n_solves=1)
