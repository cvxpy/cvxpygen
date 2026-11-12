"""
Copyright 2025-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import shutil

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
from itertools import product
from cvxpy.utilities import key_utils as ku

from cvxpygen import utils
from cvxpygen.solvers import SolverInterface, QPCanonMixin
from cvxpygen.mappings import WorkspacePointerInfo


class PDAQPInterface(QPCanonMixin, SolverInterface):
    """
    Solver interface for PDAQP, the explicit (piecewise-affine lookup-table) QP solver.

    Uses the same OSQP canonical form (P, q, d, A, l, u) since PDAQP solves the
    same parametric QP.  The :meth:`generate_code` method runs the offline MPQP
    solve and writes the lookup-table C files instead of embedding an iterative solver.
    """

    solver_name = 'PDAQP'
    cvxpy_solver_name = 'OSQP'
    stgs = {}
    parameter_update_structure = {}

    # Header / CMake info (utils.py hard-codes "pdaqp.h" for explicit mode,
    # but these are kept for completeness and write_canon_cmake compatibility).
    header_files = ['"pdaqp.h"']
    cmake_headers = ['${CMAKE_CURRENT_SOURCE_DIR}/include/pdaqp.h']
    cmake_sources = ['${CMAKE_CURRENT_SOURCE_DIR}/src/pdaqp.c']
    has_qdldl = False

    inmemory_preconditioning = False

    # The explicit solver writes sol_x / sol_y as plain C arrays (not inside an
    # OSQP workspace struct), matching the OSQP layout that utils.py expects when
    # ws_statically_allocated_in_solver_code = True.
    ws_statically_allocated_in_solver_code = True
    ws_ptrs = WorkspacePointerInfo(
        objective_value='obj_val',
        iterations='0',
        status='0',
        primal_residual='0',
        dual_residual='0',
        primal_solution='sol_x',
        dual_solution='sol_{dual_var_name}',
    )

    sol_statically_allocated = True
    status_is_int = True
    numeric_types = {'float': 'float', 'int': 'int'}

    stgs_requires_extra_struct_type = False
    stgs_dynamically_allocated = False
    stgs_reset_function = None
    stgs_direct_write_ptr = None

    docu = 'https://github.com/darnstrom/pdaqp'

    def __init__(self, canon_maps, enable_settings):
        super().__init__(canon_maps, enable_settings)
        # Set by Canonicalizer._extract() after construction
        self._problem = None
        self._solver_opts = None

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      canon, gradient, prefix) -> None:
        args = '{prefix}cpg_theta, sol_x' + (', sol_y' if configuration.explicit == 2 else '')
        self.solve_function_call = f'pdaqp_evaluate({args})'


        # set maximum number of regions and maximum number of floating point numbers
        max_regions = self._solver_opts.get('max_regions', 500) if self._solver_opts else 500
        max_floats = self._solver_opts.get('max_floats', 1e6) if self._solver_opts else 1e6

        # set precision for storing explicit solution
        c_float_store = '_Float16' if self._solver_opts and (self._solver_opts.get('fp16', False) or self._solver_opts.get('FP16', False)) else 'float'

        A = canon.parameter_canon.p['A'].toarray()
        m, n = A.shape

        H = utils.symmetric_from_upper(canon.parameter_canon.p['P']).toarray() + 1e-6 * np.eye(n)

        f = np.zeros_like(canon.parameter_canon.p['q'])
        F = np.hstack([np.eye(n), np.zeros((n, m))])

        b = np.zeros_like(canon.parameter_canon.p['u'])
        B = np.hstack([np.zeros((m, n)), np.eye(m)])

        # remove any zero rows from A and corresponding rows of b and B
        A_mask = np.any(A != 0, axis=1)
        A = A[A_mask]
        b = b[A_mask]
        B = B[A_mask]

        # extract bounds on theta (thmin, thmax) and user-defined params (lower, upper)
        thmin, thmax, lower, upper = self._get_parameter_delta_bounds(self._problem, canon)

        # remap dual variable indices from canonical row space to sol_y row space
        row_remap = np.cumsum(A_mask) - 1  # canonical_row -> sol_y_row
        dvi = canon.dual_variable_info
        for name in list(dvi.name_to_offset):
            dvi.name_to_offset[name] = row_remap[dvi.name_to_offset[name]]
            vec, old_indices = dvi.name_to_indices[name]
            dvi.name_to_indices[name] = (vec, row_remap[old_indices])

        # eliminate theta components that are fixed
        th_mask = (thmin != thmax)
        f += F[:, ~th_mask] @ thmin[~th_mask]
        b += B[:, ~th_mask] @ thmin[~th_mask]
        F = F[:, th_mask]
        B = B[:, th_mask]
        thmin, thmax = thmin[th_mask], thmax[th_mask]

        # eliminate theta components that are multiplied with zero
        th_mask_multiplied_nonzero = np.any(F != 0, axis=0) | np.any(B != 0, axis=0)
        F = F[:, th_mask_multiplied_nonzero]
        B = B[:, th_mask_multiplied_nonzero]
        thmin, thmax = thmin[th_mask_multiplied_nonzero], thmax[th_mask_multiplied_nonzero]

        th_mask_resulting = th_mask.copy()
        th_mask_resulting[th_mask] = th_mask_multiplied_nonzero

        # extract indices of equality constraints
        eq_m = canon.parameter_canon.p_id_to_size['l']
        eq_inds = np.arange(eq_m)[A_mask[:eq_m]]

        # print dimensions info
        sys.stdout.write(f'Generating explicit solver for (canonicalized) parametric QP with\n'
                         f'{len(f)} variables,\n'
                         f'{len(eq_inds)} linear equality constraints,\n'
                         f'{len(b)-len(eq_inds)} linear inequality constraints, and\n'
                         f'{len(thmin)} parameters ...\n')

        # extract variables to store
        all_names = [name for name in canon.prim_variable_info.name_to_offset]
        stored_vars = self._solver_opts.get('stored_vars', None) if self._solver_opts else None
        names_and_inds = []
        if stored_vars is not None:
            for s in stored_vars:
                v = s.variables()[0]
                sl = s.get_data()
                if sl is None:  # Variable => store all
                    names_and_inds.append((v.name(), None))
                elif len(v.shape) == 1:
                    names_and_inds.append((v.name(), sl[0]))
                else:
                    sl = [ku.format_slice(key, sh, len(v.shape)) if not ku.is_special_slice(key) else key
                          for key, sh in zip(sl[0], v.shape)]
                    ranges = [np.arange(s.start, s.stop, s.step) if type(s) == slice else s for s in sl]
                    inds = [np.ravel_multi_index(id, v.shape, order='F') for id in product(*ranges)]
                    names_and_inds.append((v.name(), inds))
        else:  # by default, store all variables
            names_and_inds = [(name, None) for name in all_names]

        shift = 0
        out_inds = np.empty(0, dtype=int)
        added_names = []
        canon.prim_variable_info.name_to_size_reduced = canon.prim_variable_info.name_to_size.copy()
        for name, inds in names_and_inds:
            offset = canon.prim_variable_info.name_to_offset.get(name, None)
            if offset is not None:
                size_orig = canon.prim_variable_info.name_to_size[name]
                inds = np.array(inds, dtype='int') if inds else np.arange(size_orig)
                size = inds.size
                if size < size_orig:
                    canon.prim_variable_info.reduced = True
                canon.prim_variable_info.name_to_size_reduced[name] = size

                out_inds = np.append(out_inds, offset + inds)

                canon.prim_variable_info.name_to_offset[name] = shift
                if size == 1:
                    canon.prim_variable_info.name_to_indices[name] = np.array([shift])
                else:
                    canon.prim_variable_info.name_to_indices[name] = np.full(size_orig, -1)
                    canon.prim_variable_info.name_to_indices[name][inds] = np.arange(size)
                added_names.append(name)
                shift += size
            # TODO: Catch case when variable does not exist

        # Remove non-stored variables from canonicalization
        for i, name in enumerate(all_names):
            if name not in added_names:
                del canon.prim_variable_info.name_to_offset[name]
                del canon.prim_variable_info.name_to_indices[name]
                del canon.prim_variable_info.name_to_size[name]
                del canon.prim_variable_info.name_to_shape[name]
                del canon.prim_variable_info.name_to_init[name]
                del canon.prim_variable_info.name_to_sym[name]
                del canon.prim_variable_info.sym[i]
                canon.prim_variable_info.reduced = True

        # construct explicit solution
        store_dual = (configuration.explicit == 2)
        # Imported here: pdaqp starts Julia (juliacall) on import.
        from pdaqp import MPQP

        mpqp = MPQP(H, f, F, A, b, B, thmin, thmax, eq_inds=eq_inds, out_inds=out_inds)
        mpqp.solve(settings={'region_limit': max_regions, 'store_dual': store_dual})
        if str(mpqp.solution_info.status) != 'Solved':
            raise Exception(f'Could not compute explicit solution: {mpqp.solution_info.status}')

        codegen_status = mpqp.codegen(dir=solver_code_dir, max_reals=max_floats, dual=store_dual, c_float_store=c_float_store)
        if codegen_status < 0:
            raise Exception('Could not generate explicit solver. Consider increasing max_reals in solver_opts.')

        include_dir = os.path.join(solver_code_dir, 'include')
        src_dir = os.path.join(solver_code_dir, 'src')
        os.makedirs(include_dir, exist_ok=True)
        os.makedirs(src_dir, exist_ok=True)
        h_file = os.path.join(include_dir, 'pdaqp.h')
        c_file = os.path.join(src_dir, 'pdaqp.c')
        shutil.move(os.path.join(solver_code_dir, 'pdaqp.h'), h_file)
        shutil.move(os.path.join(solver_code_dir, 'pdaqp.c'), c_file)

        if configuration.gradient:
            self._patch_pdaqp_for_gradient(h_file, c_file)

        # create solver_code_dir/CMakeLists.txt
        with open(os.path.join(solver_code_dir, 'CMakeLists.txt'), 'w') as fl:
            fl.write('list (APPEND pdaqp_src ${CMAKE_CURRENT_SOURCE_DIR}/src/pdaqp.c)\n')
            fl.write('list (APPEND pdaqp_head ${CMAKE_CURRENT_SOURCE_DIR}/include/pdaqp.h)\n')
            fl.write('set (solver_src ${pdaqp_src} PARENT_SCOPE)\n')
            fl.write('set (solver_head ${pdaqp_head} PARENT_SCOPE)\n')

        canon.parameter_canon.th_mask = th_mask_resulting
        canon.parameter_canon.n_param_reduced = np.count_nonzero(th_mask_resulting)
        canon.parameter_canon.n_dual_reduced = len(b)
        canon.parameter_info.lower = lower
        canon.parameter_info.upper = upper

    @staticmethod
    def _patch_pdaqp_for_gradient(h_file, c_file):
        """Post-process generated pdaqp.c/h to expose the active critical region index."""

        # header: expose the global and array symbols needed by gradient code
        utils.read_write_file(h_file, lambda text: text.replace(
            '#endif // ifndef PDAQP_H',
            'extern int pdaqp_active_region;\n'
            'extern c_float_store pdaqp_feedbacks[];\n'
            'extern c_int pdaqp_hp_list[];\n'
            '#endif // ifndef PDAQP_H',
        ))
        
        # source: add global and record active region before the feedback evaluation
        utils.read_write_file(c_file, lambda text: text
            .replace(
                'void pdaqp_evaluate(',
                'int pdaqp_active_region = -1;\nvoid pdaqp_evaluate(',
            )
            .replace(
                '    // Leaf node reached -> evaluate affine function\n',
                '    // Leaf node reached -> evaluate affine function\n'
                '    pdaqp_active_region = id;\n',
            )
        )

    @staticmethod
    def _get_parameter_delta_bounds(problem, canon):

        parameter_info = canon.parameter_info
        parameter_canon = canon.parameter_canon

        # extract bounds on user-defined parameter deltas
        lower = -1e30 * np.ones_like(parameter_info.flat_usp)
        upper = 1e30 * np.ones_like(parameter_info.flat_usp)
        lower[-1], upper[-1] = 1, 1

        for i, constraint in enumerate(problem.constraints):

            # consider pure parameter constraints
            if not constraint.variables() and constraint.parameters():

                lhs, rhs = constraint.args

                # consider simple bounds
                if isinstance(lhs, cp.Parameter) and isinstance(rhs, cp.Constant):
                    col = parameter_info.id_to_col[lhs.id]
                    upper[col:col + lhs.size] = rhs.value
                elif isinstance(lhs, cp.Constant) and isinstance(rhs, cp.Parameter):
                    col = parameter_info.id_to_col[rhs.id]
                    lower[col:col + rhs.size] = lhs.value
                else:
                    raise ValueError('Explicit mode: Parameter constraints must be simple bounds!')

                # remove dual variables corresponding to parameter constraints
                dvi = canon.dual_variable_info
                for fields in (dvi.name_to_init, dvi.name_to_vec, dvi.name_to_offset,
                               dvi.name_to_indices, dvi.name_to_size, dvi.name_to_shape,
                               dvi.name_to_parts):
                    fields.pop(f'd{i}', None)

        # map to Delta (q, u)
        id_to_mapping = parameter_canon.p_id_to_mapping
        C_qu = sp.vstack([id_to_mapping['q'], id_to_mapping['u']])
        lower_mapped, upper_mapped = C_qu @ lower, C_qu @ upper

        return np.minimum(lower_mapped, upper_mapped), np.maximum(lower_mapped, upper_mapped), lower, upper
