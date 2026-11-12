"""
Copyright 2021-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
import shutil

import osqp

from cvxpygen.solvers import SolverInterface, QPCanonMixin
from cvxpygen import utils
from cvxpygen.mappings import WorkspacePointerInfo, UpdatePendingLogic, ParameterUpdateLogic, Setting


class OSQPInterface(QPCanonMixin, SolverInterface):
    solver_name = 'OSQP'
    cvxpy_solver_name = 'OSQP'
    parameter_update_structure = {
        'PA': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P', 'A'], '&&', ['P', 'A']),
            function_call = 'osqp_update_data_mat(&solver, {prefix}Canon_Params.P->x, 0, {prefix}Canon_Params.P->nnz, {prefix}Canon_Params.A->x, 0, {prefix}Canon_Params.A->nnz)',
        ),
        'P': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P']),
            function_call = 'osqp_update_data_mat(&solver, {prefix}Canon_Params.P->x, 0, {prefix}Canon_Params.P->nnz, OSQP_NULL, 0, 0)',
        ),
        'A': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['A']),
            function_call = 'osqp_update_data_mat(&solver, OSQP_NULL, 0, 0, {prefix}Canon_Params.A->x, 0, {prefix}Canon_Params.A->nnz)'
        ),
        'qlu': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q', 'l', 'u'], '&&', ['ql', 'qu', 'lu']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, {prefix}Canon_Params.l, {prefix}Canon_Params.u)',
        ),
        'ql': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q', 'l'], '&&', ['q', 'l']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, {prefix}Canon_Params.l, OSQP_NULL)',
        ),
        'qu': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q', 'u'], '&&', ['q', 'u']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, OSQP_NULL, {prefix}Canon_Params.u)',
        ),
        'lu': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['l', 'u'], '&&', ['l', 'u']),
            function_call = 'osqp_update_data_vec(&solver, OSQP_NULL, {prefix}Canon_Params.l, {prefix}Canon_Params.u)',
        ),
        'q': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['q']),
            function_call = 'osqp_update_data_vec(&solver, {prefix}Canon_Params.q, OSQP_NULL, OSQP_NULL)',
        ),
        'l': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['l']),
            function_call = 'osqp_update_data_vec(&solver, OSQP_NULL, {prefix}Canon_Params.l, OSQP_NULL)',
        ),
        'u': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['u']),
            function_call = 'osqp_update_data_vec(&solver, OSQP_NULL, OSQP_NULL, {prefix}Canon_Params.u)',
        )
    }
    solve_function_call = 'osqp_solve(&solver)'

    # header and source files
    header_files = ['"osqp.h"', '"workspace.h"']
    cmake_headers = [
        '${CMAKE_CURRENT_SOURCE_DIR}/*.h',
        '${CMAKE_CURRENT_SOURCE_DIR}/inc/public/*.h',
        '${CMAKE_CURRENT_SOURCE_DIR}/inc/private/*.h'
    ]
    cmake_sources = ['${OSQP_SOURCES}',  '${CMAKE_CURRENT_SOURCE_DIR}/workspace.c']
    has_qdldl = True

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_statically_allocated_in_solver_code = True
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'solver.info->obj_val',
        iterations = 'solver.info->iter',
        status = 'solver.info->status',
        primal_residual = 'solver.info->prim_res',
        dual_residual = 'solver.info->dual_res',
        primal_solution = 'sol_x',
        dual_solution = 'sol_{dual_var_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # float and integer types
    numeric_types = {'float': 'double', 'int': 'int'}

    # solver settings
    stgs_dynamically_allocated = False
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = 'solver.settings'
    stgs_reset_function = {'name': 'osqp_set_default_settings', 'ptr': 'solver.settings'}
    stgs = {
        'max_iter': Setting(type='cpg_int', default='4000'),
        'eps_abs': Setting(type='cpg_float', default='1e-3'),
        'eps_rel': Setting(type='cpg_float', default='1e-3'),
        'eps_prim_inf': Setting(type='cpg_float', default='1e-4'),
        'eps_dual_inf': Setting(type='cpg_float', default='1e-4'),
        'scaled_termination': Setting(type='cpg_int', default='0'),
        'check_termination': Setting(type='cpg_int', default='25'),
        'warm_starting': Setting(type='cpg_int', default='1', name_cvxpy='warm_start'),
        'verbose': Setting(type='cpg_int', default='0', enabled=False),
        'polishing': Setting(type='cpg_int', default='0', enabled=False),
        'polish_refine_iter': Setting(type='cpg_int', default='0', enabled=False),
        'delta': Setting(type='cpg_float', default='1e-6', enabled=False)
    }

    # docu
    docu = 'https://osqp.org/docs/codegen/python.html'

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      canon, gradient, prefix) -> None:
        
        parameter_canon = canon.parameter_canon
        
        # OSQP codegen
        osqp_obj = osqp.OSQP()
        osqp_obj.setup(P=parameter_canon.p['P'], q=parameter_canon.p['q'],
                       A=parameter_canon.p['A'], l=parameter_canon.p['l'],
                       u=parameter_canon.p['u'], **(self._solver_opts or {}))

        osqp_obj.codegen(solver_code_dir, parameters='matrices', force_rewrite=True)
        
        # copy license files
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'osqp-python', 'LICENSE'),
                        os.path.join(solver_code_dir, 'LICENSE'))

        # adjust workspace.h
        utils.read_write_file(os.path.join(solver_code_dir, 'workspace.h'),
                              lambda x: x.replace('extern OSQPSolver solver;',
                                        'extern OSQPSolver solver;\n'
                                        + f'  extern OSQPFloat sol_x[{self.n_var}];\n'
                                        + f'  extern OSQPFloat sol_y[{self.n_eq + self.n_ineq}];'))

        # modify CMakeLists.txt
        utils.read_write_file(os.path.join(solver_code_dir, 'CMakeLists.txt'),
                              lambda x: utils.cut_from_expr(x, 'add_library').replace('src', '${CMAKE_CURRENT_SOURCE_DIR}/src'))

    def cmake_context_extra(self) -> dict:
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code'
        return {
            **super().cmake_context_extra(),
            'extra_cmake_include_dirs': [sdir, sdir + '/inc/public', sdir + '/inc/private'],
            'cmake_definitions': (
                ['add_definitions(-DOSQP_ENABLE_PRINTING)']
                if 'verbose' in self.enable_settings else []
            ),
        }
        
    def special_settings(self, config) -> dict:
        return {'eps_abs': 1e-4, 'eps_rel': 1e-4} if config.gradient else {}
