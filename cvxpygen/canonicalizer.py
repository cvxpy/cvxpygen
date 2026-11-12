"""
Copyright 2021-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import copy
import platform
import warnings
from typing import List, Tuple

import numpy as np
from scipy import sparse
import cvxpy as cp
from cvxpy.problems.objective import Maximize
from cvxpy.lin_ops.lin_op import CONSTANT_ID
from cvxpy.reductions.canonical_maps import ParameterTransformKind, pack_param_vec

from cvxpygen.mappings import (
    Canon, ParameterCanon, ParameterInfo, PrimalVariableInfo, DualVariableInfo
)

from cvxpygen.solvers import SolverInterface, OSQPInterface, SCSInterface, ECOSInterface, \
    ClarabelInterface, QOCOInterface, QOCOGENInterface, PDAQPInterface, UNSUPPORTED_ON_WINDOWS


class Canonicalizer:
    """
    Transforms a cp.Problem into a Canon object (solver-ready canonical form).

    Owns all extraction logic: primal/dual variable info, parameter info,
    affine mappings, and the two-stage gradient canonicalization path.
    """

    def __init__(
        self,
        solver: str,
        solver_opts: dict = {},
        enable_settings: List[str] = [],
    ) -> None:
        self.solver = solver
        self.solver_opts = solver_opts
        self.enable_settings = enable_settings

    def canonicalize(self, problem: cp.Problem) -> Tuple[Canon, SolverInterface]:
        """
        Canonicalize problem. Returns (canon, solver_interface).
        For the two-stage gradient case, returns the merged canon.
        """
        return self._extract(problem, self.solver, self.solver_opts, self.enable_settings)

    def canonicalize_two_stage(self, problem: cp.Problem) -> Tuple[Canon, Canon, Canon, SolverInterface, SolverInterface]:
        """
        Perform two-stage canonicalization for gradient computation with non-OSQP solvers.
        Returns (merged_canon, canon_gradient, canon_solver, solver_interface, gradient_interface).
        """
        canon_gradient, gradient_interface = self._extract(problem, 'OSQP', None, [])
        osqp_problem = self._get_osqp_problem(canon_gradient)
        canon_solver, solver_interface = self._extract(
            osqp_problem, self.solver, self.solver_opts, self.enable_settings
        )
        merged = self._merge(canon_gradient, canon_solver, solver_interface)
        return merged, canon_gradient, canon_solver, solver_interface, gradient_interface

    # ── private helpers ───────────────────────────────────────────────────────
    
    def _get_interface_class(self, solver_name: str) -> "SolverInterface":
        if platform.system() == 'Windows' and solver_name.upper() in UNSUPPORTED_ON_WINDOWS:
            raise ValueError(f'{solver_name} solver currently not supported on Windows.')
        mapping = {
            'OSQP': OSQPInterface,
            'SCS': SCSInterface,
            'ECOS': ECOSInterface,
            'CLARABEL': ClarabelInterface,
            'QOCO': QOCOInterface,
            'QOCOGEN': QOCOGENInterface,
            'PDAQP': PDAQPInterface,
        }
        interface = mapping.get(solver_name.upper(), None)
        if interface is None:
            raise ValueError(f'Unsupported solver: {solver_name}.')
        return interface

    def _extract(self, problem, solver, solver_opts, enable_settings) -> Tuple[Canon, object]:
        """Build the Canon from CVXPY's canonical maps (Problem.get_canonical_maps)."""
        interface_class = self._get_interface_class(solver)
        canon_maps = problem.get_canonical_maps(
            interface_class.cvxpy_solver_name, solver_opts=solver_opts)

        if not canon_maps.parameters:
            raise ValueError('Solution does not depend on parameters. Aborting code generation.')
        self._check_parameter_transforms(problem, canon_maps)
        interface_class.check_unsupported_cones(canon_maps.cone_dims)

        solver_interface = interface_class(canon_maps, enable_settings)
        solver_interface._problem = problem
        solver_interface._solver_opts = solver_opts

        parameter_info = self._get_parameter_info(canon_maps)

        parameter_canon = ParameterCanon()
        # CVXPY compiled the objective's quadratic part into P only if the solver uses it.
        P = canon_maps.structures.get('P')
        parameter_canon.quad_obj = solver_interface.solver_type == 'quadratic' or (
            P is not None and P.indices.size > 0)
        canon_p_ids = [p_id for p_id in solver_interface.canon_p_ids
                       if p_id != 'P' or parameter_canon.quad_obj]
        for p_id in canon_p_ids:
            M, v = canon_maps.maps[p_id]
            # [M | v] maps the parameter vector with its trailing constant 1.
            mapping = sparse.hstack([M, sparse.csr_matrix(v[:, None])], format='csr')
            mapping = mapping[solver_interface.canonical_mapping_rows(p_id)]
            values = mapping @ parameter_info.flat_usp
            if p_id in canon_maps.structures:
                structure = canon_maps.structures[p_id]
                parameter_canon.p[p_id] = sparse.csc_matrix(
                    (values, structure.indices, structure.indptr), shape=structure.shape)
            else:
                parameter_canon.p[p_id] = solver_interface.augment_vector_parameter(p_id, values)
            if p_id == 'd':
                parameter_canon.nonzero_d = mapping.nnz > 0
            parameter_canon.p_id_to_mapping[p_id] = mapping
            parameter_canon.p_id_to_changes[p_id] = mapping[:, :-1].nnz > 0
            parameter_canon.p_id_to_size[p_id] = mapping.shape[0]
        parameter_canon.is_maximization = isinstance(problem.objective, Maximize)
        # The canonical parameters to update when a user parameter changes.
        col, size = canon_maps.param_id_to_col, canon_maps.param_id_to_size
        parameter_canon.user_p_name_to_canon_outdated = {
            p.name(): [p_id for p_id in canon_p_ids if parameter_canon.p_id_to_mapping[p_id]
                       [:, col[p.id]:col[p.id] + size[p.id]].nnz > 0]
            for p in canon_maps.parameters
        }

        prim_variable_info = self._primal_info_from_maps(problem, canon_maps)
        dual_variable_info = self._dual_info_from_maps(problem, canon_maps, solver_interface)
        canon = Canon(prim_variable_info, dual_variable_info, parameter_info, parameter_canon)
        return canon, solver_interface

    @staticmethod
    def _primal_info_from_maps(problem, canon_maps) -> PrimalVariableInfo:
        """Primal variables as gathers from the solver's primal vector."""
        infos = {info.var_id: info for info in canon_maps.primal_vars}
        name_to_indices = {}
        for var in problem.variables():
            indices = canon_maps.primal_recovery[var.id]
            if infos[var.id].sparse_idx is not None or not isinstance(indices, np.ndarray):
                raise ValueError(
                    f'Code generation does not support variable {var.name()} '
                    '(sparse or diagonal variables, or variables CVXPY eliminates).')
            name_to_indices[var.name()] = indices
        names = list(name_to_indices)
        sym = [infos[var.id].is_symmetric for var in problem.variables()]
        return PrimalVariableInfo(
            {n: int(idx[0]) if idx.size else 0 for n, idx in name_to_indices.items()},
            name_to_indices,
            {var.name(): var.size for var in problem.variables()},
            {var.name(): var.shape for var in problem.variables()},
            {var.name(): 0 if len(var.shape) == 0 else np.zeros(var.shape)
             for var in problem.variables()},
            dict(zip(names, sym)), sym,
        )

    @staticmethod
    def _dual_info_from_maps(problem, canon_maps, solver_interface) -> DualVariableInfo:
        """Constraint duals as contiguous blocks of the solver's dual vector.

        CVXPY recovers each dual variable separately; code generation exposes
        one block per constraint (named d<i> after its position in
        problem.constraints), which its dual variables tile.
        """
        infos = {info.dual_var_id: info for info in canon_maps.dual_vars}
        # Solvers with split dual vectors return y (equalities) and z (cones).
        n_y = solver_interface.n_eq if solver_interface.dual_var_split else np.inf
        offsets, indices, sizes, shapes, inits, vecs, parts = {}, {}, {}, {}, {}, {}, {}
        for i, con in enumerate(problem.constraints):
            recs = [canon_maps.dual_recovery[dv.id] for dv in con.dual_variables]
            idx = np.sort(np.concatenate(recs)) if all(
                isinstance(r, np.ndarray) for r in recs) else None
            if idx is None or idx.size == 0 or np.any(np.diff(idx) != 1) \
                    or (idx[0] < n_y <= idx[-1]):
                warnings.warn(f'The dual of constraint {i} ({type(con).__name__}) is not a '
                              'block of the solver solution and is not generated.')
                continue
            name = f'd{i}'
            start = int(idx[0])
            vec_index = 0 if start < n_y else 1
            offset = start - (0 if vec_index == 0 else n_y)
            vecs[name] = solver_interface.dual_var_names[vec_index]
            offsets[name] = offset
            indices[name] = (vecs[name], idx - start + offset)
            sizes[name] = idx.size
            shapes[name] = con.shape if np.prod(con.shape) == idx.size else None
            inits[name] = 0 if idx.size == 1 else np.zeros(idx.size)
            parts[name] = [(j, recs[j] - start, infos[dv.id].shape)
                           for j, dv in enumerate(con.dual_variables)]
        return DualVariableInfo(offsets, indices, sizes, shapes, inits, vecs, parts)

    @staticmethod
    def _check_parameter_transforms(problem, canon_maps) -> None:
        """Reject parameters whose compiled form code generation cannot reproduce.

        Code generation takes the compiled parameters as its parameters. That
        matches the user's parameters except for the reductions' parameter
        transforms, of which only the gathers of the stored entries of diagonal
        and sparse parameters are supported (see get_param_value in the
        generated Python wrapper).
        """
        user_params = {p.id: p for p in problem.parameters()}
        for transform in canon_maps.param_transforms:
            param = user_params.get(transform.source_id)
            if param is None or transform.kind != ParameterTransformKind.GATHER or \
                    not (param.attributes['diag'] or param.attributes['sparsity']):
                name = param.name() if param is not None else transform.source_id
                raise ValueError(
                    f'Code generation does not support parameter {name}: only dense, '
                    'diagonal and sparse real parameters are supported (no symmetric, '
                    'PSD, NSD, hermitian or complex parameters).')

    @staticmethod
    def _get_parameter_info(canon_maps) -> ParameterInfo:
        parameters = canon_maps.parameters
        user_p_num = len(parameters)
        user_p_names = [par.name() for par in parameters]
        # The constant column follows the parameters in the canonical mappings.
        user_p_id_to_col = {**canon_maps.param_id_to_col,
                            CONSTANT_ID: canon_maps.total_param_size}
        user_p_ids = list(user_p_id_to_col.keys())
        user_p_id_to_size = canon_maps.param_id_to_size
        user_p_id_to_param = {par.id: par for par in parameters}

        user_p_name_to_shape = {
            user_p_id_to_param[p_id].name(): user_p_id_to_param[p_id].shape
            for p_id in user_p_id_to_size.keys()
        }
        user_p_name_to_size_usp = {
            user_p_id_to_param[p_id].name(): size
            for p_id, size in user_p_id_to_size.items()
        }
        user_p_col_to_name_usp = {}
        cum_sum = 0
        for name, size in user_p_name_to_size_usp.items():
            user_p_col_to_name_usp[cum_sum] = name
            cum_sum += size

        user_p_writable = {}
        for p_name, p in zip(user_p_names, parameters):
            if p.value is None:
                p.project_and_assign(np.ones(p.shape))
                if type(p.value) in [sparse.dia_matrix, sparse.dia_array]:
                    p.value = p.value.toarray()
            if len(p.shape) < 2:
                user_p_writable[p_name] = p.value
            else:
                user_p_writable[p_name] = p.value.flatten(order='F')

        # The packed compiled parameters, followed by the constant 1.
        user_p_flat_usp = np.append(pack_param_vec(
            {p.id: p.value for p in parameters}, canon_maps.param_id_to_col,
            user_p_id_to_size, canon_maps.total_param_size), 1.0)

        return ParameterInfo(
            user_p_col_to_name_usp, user_p_flat_usp, user_p_id_to_col,
            user_p_ids, user_p_name_to_shape, user_p_name_to_size_usp,
            user_p_names, user_p_num, user_p_writable, None, None,
        )

    def _get_osqp_problem(self, canon_osqp) -> cp.Problem:
        p = canon_osqp.parameter_canon.p
        n_eq = canon_osqp.parameter_canon.p_id_to_size['l']

        if canon_osqp.parameter_canon.p_id_to_changes['P']:
            raise ValueError(
                'Problem does not follow extended DPP rules for differentiation with general solvers '
                '(other than OSQP). Quadratics cannot be multiplied with parameters.'
            )

        row, col = p['P'].nonzero()
        assert all(row == col)

        osqp_x = cp.Variable(p['q'].shape, name='osqp_x')
        osqp_q = cp.Parameter(p['q'].shape, name='osqp_q')
        osqp_A = cp.Parameter(p['A'].shape, name='osqp_A', sparsity=p['A'].nonzero())
        osqp_l = cp.Parameter((n_eq,), name='osqp_l')
        osqp_u = cp.Parameter(p['u'].shape, name='osqp_u')
        osqp_P_diag_sqrt = np.sqrt(np.diag(p['P'].todense()))

        return cp.Problem(
            cp.Minimize(
                0.5 * cp.sum_squares(cp.multiply(osqp_x, osqp_P_diag_sqrt)) + osqp_q @ osqp_x
            ),
            [osqp_l <= osqp_A[:n_eq, :] @ osqp_x, osqp_A @ osqp_x <= osqp_u],
        )

    def _merge(self, canon_first, canon_second, solver_interface) -> Canon:
        pc_first = canon_first.parameter_canon
        pc_second = canon_second.parameter_canon
        n_user_param = canon_first.parameter_info.flat_usp.size
        ret_prim_func_needed = solver_interface.ret_prim_func_exists(
            canon_second.prim_variable_info
        )

        pc = ParameterCanon()
        pc.is_maximization = pc_first.is_maximization
        pc.nonzero_d = pc_first.nonzero_d
        pc.quad_obj = pc_second.quad_obj
        pc.p = pc_second.p
        pc.p_id_to_size = pc_second.p_id_to_size
        pc.p_id_to_changes = pc_second.p_id_to_changes

        pc.p_id_to_mapping = {}
        map_first = []
        for col in sorted(canon_second.parameter_info.col_to_name_usp):
            name = canon_second.parameter_info.col_to_name_usp[col][5:]
            map_first.append(pc_first.p_id_to_mapping[name])
        map_first.append(self._create_constant_map(1, n_user_param, 1.))
        map_first = sparse.vstack(map_first)
        for p_id, map_second in pc_second.p_id_to_mapping.items():
            pc.p_id_to_mapping[p_id] = map_second @ map_first

        pc.user_p_name_to_canon_outdated = {}
        for user_p_name, canon_p_ids in pc_first.user_p_name_to_canon_outdated.items():
            canon_outdated = []
            for p_id in canon_p_ids:
                canon_outdated.extend(
                    pc_second.user_p_name_to_canon_outdated[f'osqp_{p_id}']
                )
            pc.user_p_name_to_canon_outdated[user_p_name] = list(set(canon_outdated))

        pvi = copy.deepcopy(canon_first.prim_variable_info)
        if not ret_prim_func_needed:
            offset_second = canon_second.prim_variable_info.name_to_offset['osqp_x']
            for name in pvi.name_to_offset.keys():
                pvi.name_to_offset[name] += offset_second
                pvi.name_to_indices[name] += offset_second

        return Canon(pvi, canon_first.dual_variable_info, canon_first.parameter_info, pc)

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _create_constant_map(n, m, val):
        data = np.full(n, val)
        rows = np.arange(n)
        cols = np.full(n, m - 1)
        return sparse.csc_matrix((data, (rows, cols)), shape=(n, m))
