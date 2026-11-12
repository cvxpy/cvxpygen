"""
Copyright 2025-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
import shutil

import numpy as np

from cvxpygen.solvers import SolverInterface
from cvxpygen import utils
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, \
    ParameterCanon, WorkspacePointerInfo, UpdatePendingLogic, ParameterUpdateLogic, Setting
    

class QOCOGENInterface(SolverInterface):
    solver_name = 'QOCOGEN'
    cvxpy_solver_name = 'QOCO'
    solver_type = 'conic'
    canon_p_ids = ['P', 'c', 'd', 'A', 'b', 'G', 'h']
    solve_function_call = 'qoco_custom_solve(&{prefix}qoco_custom_workspace)'

    # header files
    header_files = ['"qoco_custom.h"']
    cmake_headers = ['${qoco_custom_headers}']
    cmake_sources = ['${qoco_custom_sources}']
    has_qdldl = False

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'qoco_custom_workspace.sol.obj',
        iterations = 'qoco_custom_workspace.sol.iters',
        status = 'qoco_custom_workspace.sol.status',
        primal_residual = 'qoco_custom_workspace.sol.pres',
        dual_residual = 'qoco_custom_workspace.sol.dres',
        primal_solution = 'qoco_custom_workspace.sol.x',
        dual_solution = 'qoco_custom_workspace.sol.{dual_var_name}',
        settings = 'qoco_custom_workspace.settings.{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'double', 'int': 'int'}

    # solver settings
    stgs_dynamically_allocated = False
    stgs_requires_extra_struct_type = False
    stgs_direct_write_ptr = '(&({prefix}qoco_custom_workspace.settings))'
    stgs_translation = "{}"
    stgs_reset_function = {'name': 'set_default_settings', 'ptr': '&{prefix}qoco_custom_workspace'}
    stgs = {
        'max_iters': Setting(type='cpg_int', default='200'),
        'bisect_iters': Setting(type='cpg_int', default='5'),
        'ruiz_iters': Setting(type='cpg_int', default='0'),
        'iter_ref_iters': Setting(type='cpg_int', default='1'),
        'kkt_static_reg': Setting(type='cpg_float', default='1e-8'),
        'kkt_dynamic_reg': Setting(type='cpg_float', default='1e-8'),
        'abstol': Setting(type='cpg_float', default='1e-7'),
        'reltol': Setting(type='cpg_float', default='1e-7'),
        'abstol_inacc': Setting(type='cpg_float', default='1e-5'),
        'reltol_inacc': Setting(type='cpg_float', default='1e-5'),
        'verbose': Setting(type='cpg_int', default='0')
    }
    
    dual_var_split = True
    dual_var_names = ['y', 'z']

    # docu
    docu = 'https://qoco-org.github.io/qoco/'

    def __init__(self, canon_maps, enable_settings):
        cone_dims = canon_maps.cone_dims
        n_var = canon_maps.maps['c'][1].size
        n_eq = cone_dims.zero
        n_ineq = canon_maps.structures['G'].shape[0]
        q = np.array(cone_dims.soc)
        has_P = 'P' in canon_maps.structures

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq,
                           'l': cone_dims.nonneg,
                           'nsoc': len(cone_dims.soc),
                           'q': q}

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition='{prefix}qoco_custom_workspace.n <= 0'),
                function_call=f'load_data(&{{prefix}}qoco_custom_workspace)'
            ),
            'A': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['A']),
                function_call='update_A(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.A->x)'
            ),
            'G': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['G']),
                function_call='update_G(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.G->x)'
            ),
            'c': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['c']),
                function_call = 'update_c(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.c)'
            ),
            'b': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['b']),
                function_call = 'update_b(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.b)'
            ),
            'h': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['h']),
                function_call = 'update_h(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.h)'
            ),
        }
        if has_P:
            self.parameter_update_structure['P'] = ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['P']),
                function_call = 'update_P(&{prefix}qoco_custom_workspace, {prefix}Canon_Params.P->x)'
            )

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0:
            raise ValueError(
                'Exponential cones is not supported for QOCOGEN.')

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      canon, gradient, prefix) -> None:
        import qocogen

        # Need to check if problem has quadratic objective.
        parameter_canon = canon.parameter_canon
        P = parameter_canon.p['P'] if 'P' in parameter_canon.p.keys() else None

        # Generate qoco_custom
        qocogen.generate_solver(self.canon_constants['n'], self.canon_constants['m'], self.canon_constants['p'], P, 
                                parameter_canon.p['c'], parameter_canon.p['A'], parameter_canon.p['b'], parameter_canon.p['G'], parameter_canon.p['h'],
                                self.canon_constants['l'], self.canon_constants['nsoc'], self.canon_constants['q'],
                                os.path.join(code_dir, 'c'), "solver_code")
        
        # Copy LICENSE
        shutil.copyfile(os.path.join(code_dir, 'c', 'solver_code', 'LICENSE'),
                        os.path.join(code_dir, 'LICENSE'))

    def cmake_context_extra(self) -> dict:
        return {
            **super().cmake_context_extra(),
            'solver_code_cmake_include_dir': '${CMAKE_CURRENT_SOURCE_DIR}/solver_code',
        }

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// qoco_custom array of SOC dimensions\n')
            utils.write_vec_prot(f, self.canon_constants['q'], f'{prefix}qoco_custom_q', 'cpg_int')
        f.write('\n// qoco_custom workspace\n')
        f.write(f'extern Workspace {prefix}qoco_custom_workspace;\n')
        f.write('\n// qoco_custom exit flag\n')
        f.write(f'extern cpg_int {prefix}qoco_custom_flag;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// qoco_custom array of SOC dimensions\n')
            utils.write_vec_def(f, self.canon_constants['q'], f'{prefix}qoco_custom_q', 'cpg_int')
        f.write('\n// qoco_custom workspace\n')
        f.write(f'Workspace {prefix}qoco_custom_workspace;\n')
        f.write('\n// qoco_custom exit flag\n')
        f.write(f'cpg_int {prefix}qoco_custom_flag = -99;\n')

class QOCOInterface(SolverInterface):
    solver_name = 'QOCO'
    cvxpy_solver_name = 'QOCO'
    solver_type = 'conic'
    canon_p_ids = ['P', 'c', 'd', 'A', 'b', 'G', 'h']
    solve_function_call = 'qoco_solve({prefix}qoco_solver)'

    # header files
    header_files = ['"qoco.h"']
    cmake_headers = ['${qoco_headers}']
    cmake_sources = ['${qoco_sources}']
    has_qdldl = False

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'qoco_solver->sol->obj',
        iterations = 'qoco_solver->sol->iters',
        status = 'qoco_solver->sol->status',
        primal_residual = 'qoco_solver->sol->pres',
        dual_residual = 'qoco_solver->sol->dres',
        primal_solution = 'qoco_solver->sol->x',
        dual_solution = 'qoco_solver->sol->{dual_var_name}',
        settings = 'qoco_solver->settings->{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'QOCOFloat', 'int': 'QOCOInt'}

    # solver settings
    stgs_dynamically_allocated = True
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = None
    stgs_reset_function = None
    stgs = {
        'max_iters': Setting(type='cpg_int', default='200'),
        'bisect_iters': Setting(type='cpg_int', default='5'),
        'ruiz_iters': Setting(type='cpg_int', default='0'),
        'iter_ref_iters': Setting(type='cpg_int', default='1'),
        'kkt_static_reg': Setting(type='cpg_float', default='1e-8'),
        'kkt_dynamic_reg': Setting(type='cpg_float', default='1e-8'),
        'abstol': Setting(type='cpg_float', default='1e-7'),
        'reltol': Setting(type='cpg_float', default='1e-7'),
        'abstol_inacc': Setting(type='cpg_float', default='1e-5'),
        'reltol_inacc': Setting(type='cpg_float', default='1e-5'),
        'verbose': Setting(type='cpg_int', default='0')
    }

    # dual variables split into y and z vectors
    dual_var_split = True
    dual_var_names = ['y', 'z']

    # docu
    docu = 'https://qoco-org.github.io/qoco/'

    def __init__(self, canon_maps, enable_settings):
        cone_dims = canon_maps.cone_dims
        n_var = canon_maps.maps['c'][1].size
        n_eq = cone_dims.zero
        n_ineq = canon_maps.structures['G'].shape[0]
        q = np.array(cone_dims.soc)
        has_P = 'P' in canon_maps.structures

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq,
                           'l': cone_dims.nonneg,
                           'nsoc': len(cone_dims.soc),
                           'q': q}

        function_call = (
            f'{{prefix}}cpg_copy_all();\n'
            '   QOCOCscMatrix* P = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n'
            '   QOCOCscMatrix* A = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n'
            '   QOCOCscMatrix* G = (QOCOCscMatrix*)malloc(sizeof(QOCOCscMatrix));\n'
        )
        if has_P:
            function_call += (
                f'   qoco_set_csc(P, {canon_constants["n"]}, {canon_constants["n"]}, '
                f'{{prefix}}Canon_Params.P->nnz, {{prefix}}Canon_Params.P->x, '
                f'{{prefix}}Canon_Params.P->p, {{prefix}}Canon_Params.P->i);\n'
            )
        else:
            function_call += '   P = NULL;\n'

        if canon_constants["p"] > 0:
            function_call += (
                f'   qoco_set_csc(A, {canon_constants["p"]}, {canon_constants["n"]}, '
                f'{{prefix}}Canon_Params.A->nnz, {{prefix}}Canon_Params.A->x, '
                f'{{prefix}}Canon_Params.A->p, {{prefix}}Canon_Params.A->i);\n'
            )
        else:
            function_call += '   A = NULL;\n'

        if canon_constants["m"] > 0:
            function_call += (
                f'   qoco_set_csc(G, {canon_constants["m"]}, {canon_constants["n"]}, '
                f'{{prefix}}Canon_Params.G->nnz, {{prefix}}Canon_Params.G->x, '
                f'{{prefix}}Canon_Params.G->p, {{prefix}}Canon_Params.G->i);\n'
            )
        else:
            function_call += '   G = NULL;\n'

        function_call += (
            '   QOCOSettings* qoco_settings = (QOCOSettings*)malloc(sizeof(QOCOSettings));\n'
            '   set_default_settings(qoco_settings);\n'
            f'   {{prefix}}qoco_solver = (QOCOSolver*)malloc(sizeof(QOCOSolver));\n'
            f'   qoco_setup({{prefix}}qoco_solver, {canon_constants["n"]}, {canon_constants["m"]}, {canon_constants["p"]}, P, '
            f'{{prefix}}Canon_Params.c, A, '
            f'{"NULL" if canon_constants["p"] == 0 else f"{{prefix}}Canon_Params.b"}, '
            f'G, {"NULL" if canon_constants["m"] == 0 else f"{{prefix}}Canon_Params.h"}, '
            f'{canon_constants["l"]}, {canon_constants["nsoc"]}, '
            f'{"NULL" if canon_constants["nsoc"] == 0 else f"(int *) &{{prefix}}qoco_q"}, '
            'qoco_settings)'
        )

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition='!{prefix}qoco_solver'),
                function_call=function_call
            ),
            'A': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['A']),
                function_call='update_matrix_data({prefix}qoco_solver, NULL, {prefix}Canon_Params.A->x, NULL)'
            ),
            'G': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['G']),
                function_call='update_matrix_data({prefix}qoco_solver, NULL, NULL, {prefix}Canon_Params.G->x)'
            ),
            'c': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['c']),
                function_call = 'update_vector_data({prefix}qoco_solver, {prefix}Canon_Params.c, NULL, NULL)'
            ),
            'b': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['b']),
                function_call = 'update_vector_data({prefix}qoco_solver, NULL, {prefix}Canon_Params.b, NULL)'
            ),
            'h': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['h']),
                function_call = 'update_vector_data({prefix}qoco_solver, NULL, NULL, {prefix}Canon_Params.h)'
            ),
        }
        if has_P:
            self.parameter_update_structure['P'] = ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['P']),
                function_call = 'update_matrix_data({prefix}qoco_solver, {prefix}Canon_Params.P->x, NULL, NULL)'
            )

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0:
            raise ValueError(
                'QOCO does not support exponential cones.')

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                  parameter_canon: ParameterCanon, gradient, prefix) -> None:

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'lib', 'configure', 'algebra']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'qoco', dtc),
                            os.path.join(solver_code_dir, dtc))
        
        files_to_copy = ['CMakeLists.txt', 'LICENSE']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'qoco', fl),
                            os.path.join(solver_code_dir, fl))
        
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'qoco', 'LICENSE'),
                        os.path.join(code_dir, 'LICENSE'))

    def cmake_context_extra(self) -> dict:
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        return {
            **super().cmake_context_extra(),
            'extra_cmake_include_dirs': [sdir + 'lib/amd', sdir + 'lib/qdldl/include'],
            'cmake_target_link_libs': ['qocostatic'],
        }

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// QOCO array of SOC dimensions\n')
            utils.write_vec_prot(f, self.canon_constants['q'], f'{prefix}qoco_q', 'cpg_int')
        f.write('\n// QOCO workspace\n')
        f.write(f'extern QOCOSolver* {prefix}qoco_solver;\n')
        f.write('\n// QOCO exit flag\n')
        f.write(f'extern cpg_int {prefix}qoco_flag;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['nsoc'] > 0:
            f.write('\n// QOCO array of SOC dimensions\n')
            utils.write_vec_def(f, self.canon_constants['q'], f'{prefix}qoco_q', 'cpg_int')
        f.write('\n// QOCO solver\n')
        f.write(f'QOCOSolver* {prefix}qoco_solver = NULL;\n')
        f.write('\n// QOCO exit flag\n')
        f.write(f'cpg_int {prefix}qoco_flag = -99;\n')
