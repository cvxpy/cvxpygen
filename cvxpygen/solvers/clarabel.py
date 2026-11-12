"""
Copyright 2023-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import shutil

import numpy as np

from cvxpygen.solvers import SolverInterface
from cvxpygen import utils
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, \
    WorkspacePointerInfo, UpdatePendingLogic, ParameterUpdateLogic, Setting
    

class ClarabelInterface(SolverInterface):
    solver_name = 'Clarabel'
    cvxpy_solver_name = 'CLARABEL'
    solver_type = 'conic'
    canon_p_ids = ['P', 'q', 'd', 'A', 'b']
    gradient_supported = False

    # header and source files
    header_files = ['<Clarabel>']
    cmake_headers, cmake_sources = [], []
    has_qdldl = False

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'solution.obj_val',
        iterations = 'solution.iterations',
        status = 'solution.status',
        primal_residual = 'solution.r_prim',
        dual_residual = 'solution.r_dual',
        primal_solution = 'solution.x',
        dual_solution = 'solution.{dual_var_name}',
        settings = 'settings.{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'ClarabelFloat', 'int': 'uintptr_t'}
    
    # solver settings
    stgs_dynamically_allocated = True
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = None
    stgs_reset_function = None
    # Comprehensive Clarabel settings (matched to Clarabel v0.6+ defaults)
    stgs = {
        # Main algorithm settings
        'max_iter': Setting(type='cpg_int', default='200', name_cvxpy='max_iters'),
        'time_limit': Setting(type='cpg_float', default='1e10'),  # Inf not supported, use very large number
        'verbose': Setting(type='cpg_int', default='1'),
        'max_step_fraction': Setting(type='cpg_float', default='0.99'),

        # Full accuracy tolerance settings
        'tol_gap_abs': Setting(type='cpg_float', default='1e-8'),
        'tol_gap_rel': Setting(type='cpg_float', default='1e-8'),
        'tol_feas': Setting(type='cpg_float', default='1e-8'),
        'tol_infeas_abs': Setting(type='cpg_float', default='1e-8'),
        'tol_infeas_rel': Setting(type='cpg_float', default='1e-8'),
        'tol_ktratio': Setting(type='cpg_float', default='1e-6'),

        # Reduced accuracy tolerance settings
        'reduced_tol_gap_abs': Setting(type='cpg_float', default='5e-5'),
        'reduced_tol_gap_rel': Setting(type='cpg_float', default='5e-5'),
        'reduced_tol_feas': Setting(type='cpg_float', default='1e-4'),
        'reduced_tol_infeas_abs': Setting(type='cpg_float', default='5e-5'),
        'reduced_tol_infeas_rel': Setting(type='cpg_float', default='5e-5'),
        'reduced_tol_ktratio': Setting(type='cpg_float', default='1e-4'),

        # Equilibration settings
        'equilibrate_enable': Setting(type='cpg_int', default='1'),
        'equilibrate_max_iter': Setting(type='cpg_int', default='10'),
        'equilibrate_min_scaling': Setting(type='cpg_float', default='1e-4'),
        'equilibrate_max_scaling': Setting(type='cpg_float', default='1e4'),

        # Step size settings
        'linesearch_backtrack_step': Setting(type='cpg_float', default='0.8'),
        'min_switch_step_length': Setting(type='cpg_float', default='0.1'),
        'min_terminate_step_length': Setting(type='cpg_float', default='1e-4'),

        # Linear solver settings
        'direct_kkt_solver': Setting(type='cpg_int', default='1'),
        # Note: direct_solve_method is an enum in C API, not easily configurable
        # Note: max_threads not available in Clarabel C API

        # Regularization settings
        'static_regularization_enable': Setting(type='cpg_int', default='1'),
        'static_regularization_constant': Setting(type='cpg_float', default='1e-8'),
        'static_regularization_proportional': Setting(type='cpg_float', default='2.2e-16'),
        'dynamic_regularization_enable': Setting(type='cpg_int', default='1'),
        'dynamic_regularization_eps': Setting(type='cpg_float', default='1e-13'),
        'dynamic_regularization_delta': Setting(type='cpg_float', default='2e-7'),

        # Iterative refinement
        'iterative_refinement_enable': Setting(type='cpg_int', default='1'),
        'iterative_refinement_reltol': Setting(type='cpg_float', default='1e-13'),
        'iterative_refinement_abstol': Setting(type='cpg_float', default='1e-12'),
        'iterative_refinement_max_iter': Setting(type='cpg_int', default='10'),
        'iterative_refinement_stop_ratio': Setting(type='cpg_float', default='5.0'),

        # Presolve
        'presolve_enable': Setting(type='cpg_int', default='1')
    }

    # dual variables split into two vectors
    dual_var_split = False
    dual_var_names = ['z']

    # docu
    docu = 'https://oxfordcontrol.github.io/ClarabelDocs/'

    def __init__(self, canon_maps, enable_settings):
        n_eq, n_var = canon_maps.structures['A'].shape
        n_ineq = 0
        cone_dims = canon_maps.cone_dims

        canon_constants = {'n': n_var, 'm': n_eq,
                           'cone_dims_zero': cone_dims.zero,
                           'cone_dims_nonneg': cone_dims.nonneg,
                           'cone_dims_exp': cone_dims.exp,
                           'cone_dims_soc': np.array(cone_dims.soc),
                           'cone_dims_psd': np.array(cone_dims.psd),
                           'cone_dims_p3d': np.array(cone_dims.p3d)}
        
        canon_constants['n_cone_types'] = int(cone_dims.zero > 0) + \
            int(cone_dims.nonneg > 0) + \
            int(cone_dims.exp > 0) + \
            len(cone_dims.soc) + \
            len(cone_dims.psd) + \
            len(cone_dims.p3d)
        
        canon_constants['cd_to_t'] = {
            'cone_dims_zero': 'ClarabelZeroConeT',
            'cone_dims_nonneg': 'ClarabelNonnegativeConeT',
            'cone_dims_exp': 'ClarabelExponentialConeT',
            'cone_dims_soc': 'ClarabelSecondOrderConeT',
            'cone_dims_psd': 'ClarabelPSDTriangleConeT',
            'cone_dims_p3d': 'ClarabelPowerConeT'
        }

        # catch LP case (hack, until Clarabel permits passing a zero pointer as &P)
        is_lp = canon_maps.structures['P'].indices.size == 0
        if is_lp:
            extra_condition = '1'
            P_p = f'(cpg_int[]){{{{ {", ".join(["0"]*(n_var+1))} }}}}'
            P_i = '0'
            P_x = '0'
            update_after_init = ['A', 'q', 'b']
        else:
            extra_condition = '!{prefix}solver'
            P_p = '{prefix}Canon_Params_conditioning.P->p'
            P_i = '{prefix}Canon_Params_conditioning.P->i'
            P_x = '{prefix}Canon_Params_conditioning.P->x'
            update_after_init = ['P', 'A', 'q', 'b']

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition=extra_condition, functions_if_false=update_after_init),
                function_call= \
                    f'{{prefix}}cpg_copy_all();\n'
                    f'    clarabel_CscMatrix_init(&{{prefix}}P, {canon_constants["n"]}, {canon_constants["n"]}, {P_p}, {P_i}, {P_x});\n'
                    f'    clarabel_CscMatrix_init(&{{prefix}}A, {canon_constants["m"]}, {canon_constants["n"]}, {{prefix}}Canon_Params_conditioning.A->p, {{prefix}}Canon_Params_conditioning.A->i, {{prefix}}Canon_Params_conditioning.A->x);\n' \
                    f'    {{prefix}}settings = clarabel_DefaultSettings_default()'
            ),
            'A': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['A']),
                function_call = f'{{prefix}}cpg_copy_A();\n      clarabel_CscMatrix_init(&{{prefix}}A, {canon_constants["m"]}, {canon_constants["n"]}, {{prefix}}Canon_Params_conditioning.A->p, {{prefix}}Canon_Params_conditioning.A->i, {{prefix}}Canon_Params_conditioning.A->x)'
            ),
            'q': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['q']),
                function_call = f'{{prefix}}cpg_copy_q()'
            ),
            'b': ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['b']),
                function_call = f'{{prefix}}cpg_copy_b()'
            ),
        }
        
        if not is_lp:
            self.parameter_update_structure['P'] = ParameterUpdateLogic(
                update_pending_logic = UpdatePendingLogic(['P']),
                function_call = f'{{prefix}}cpg_copy_P();\n      clarabel_CscMatrix_init(&{{prefix}}P, {canon_constants["n"]}, {canon_constants["n"]}, {P_p}, {P_i}, {P_x})'
            )

        self.solve_function_call = \
            f'{{prefix}}solver = clarabel_DefaultSolver_new(&{{prefix}}P, {{prefix}}Canon_Params_conditioning.q, &{{prefix}}A, {{prefix}}Canon_Params_conditioning.b, {canon_constants["n_cone_types"]}, {{prefix}}cones, &{{prefix}}settings);\n' \
            f'  clarabel_DefaultSolver_solve({{prefix}}solver);\n' \
            f'  {{prefix}}solution = clarabel_DefaultSolver_solution({{prefix}}solver)'

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, canon_constants, enable_settings)

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      canon, gradient, prefix) -> None:

        # check if sdp cones are present
        is_sdp = len(self.canon_constants['cone_dims_psd']) > 0
        if is_sdp:
            sys.stdout.write('WARNING: You are generating code for an SDP with Clarabel, which requires BLAS and LAPACK within Rust-C wrappers - expect large compilation time and binary size.\n')

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['rust_wrapper', 'include', 'Clarabel.rs']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'Clarabel.cpp', dtc),
                            os.path.join(solver_code_dir, dtc))
        files_to_copy = ['CMakeLists.txt', 'LICENSE.md']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'Clarabel.cpp', fl),
                            os.path.join(solver_code_dir, fl))

        # remove examples target from Clarabel.cpp/CMakeLists.txt and set build type to Release
        replacements = [
            ('add_subdirectory(examples)', '# add_subdirectory(examples)'),
            ('set(CMAKE_C_STANDARD_REQUIRED True)', 'set(CMAKE_C_STANDARD_REQUIRED True)\n\n# set build type to Release\nset(CMAKE_BUILD_TYPE Release)')
        ]
        utils.read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                        lambda x: utils.multiple_replace(x, replacements))

        # add sdp flag
        if is_sdp:
            utils.read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'CMakeLists.txt'),
                            lambda x: x.replace('set(CLARABEL_FEATURE_SDP "none"', 'set(CLARABEL_FEATURE_SDP "sdp-openblas"'))

        # adjust Clarabel.cpp/rust_wrapper/CMakeLists.txt
        replacements = [
            ('${CMAKE_SOURCE_DIR}/', '${CMAKE_SOURCE_DIR}/c/solver_code/'),
            ('/libclarabel_c.lib', '/clarabel_c.lib'),  # until fixed on Clarabel side
            (
                'set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/c/solver_code/rust_wrapper/target/release")',
                'if (ARM64)\n'
                '        message(STATUS "ARM64 detected")\n'
                '        set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/c/solver_code/rust_wrapper/target/aarch64-apple-darwin/release")\n'
                '    else()\n'
                '        set(clarabel_c_output_directory "${CMAKE_SOURCE_DIR}/c/solver_code/rust_wrapper/target/release")\n'
                '    endif()'
            ),
            (
                '# Add the cargo project as a custom target',
                '# Add the cargo project as a custom target\n'
                'if(ARM64)\n'
                '   set(clarabel_c_build_flags "${clarabel_c_build_flags};--target;aarch64-apple-darwin")\n'
                'endif()'
            ),
            (
                'COMMAND cargo build ${clarabel_c_build_flags}',
                'BYPRODUCTS ${clarabel_c_output_directory}/libclarabel_c.a\n'
                '    COMMAND cargo build ${clarabel_c_build_flags}'
            )
        ]
        utils.read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'rust_wrapper', 'CMakeLists.txt'),
                              lambda x: utils.multiple_replace(x, replacements))

        # adjust Clarabel
        utils.read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'Clarabel'),
                              lambda x: x.replace('cpp/', 'c/'))
        
    def cmake_context_extra(self) -> dict:
        packages = []
        cmake_target_link_libs = ['libclarabel_c_static']
        if len(self.canon_constants['cone_dims_psd']) > 0:  # sdp
            packages = ['BLAS', 'LAPACK']
            cmake_target_link_libs.extend(['${BLAS_LIBRARIES}', '${LAPACK_LIBRARIES}'])
        return {
            **super().cmake_context_extra(),
            'packages': packages,
            'cmake_target_link_libs': cmake_target_link_libs,
        }

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        f.write('\n// Clarabel workspace\n')
        f.write(f'extern ClarabelCscMatrix {prefix}P;\n')
        f.write(f'extern ClarabelCscMatrix {prefix}A;\n')
        f.write(f'extern ClarabelSupportedConeT {prefix}cones[{self.canon_constants["n_cone_types"]}];\n')
        f.write(f'extern ClarabelDefaultSettings {prefix}settings;\n')
        f.write(f'extern ClarabelDefaultSolver *{prefix}solver;\n')
        f.write(f'extern ClarabelDefaultSolution {prefix}solution;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        f.write('\n// Clarabel workspace\n')
        f.write(f'ClarabelCscMatrix {prefix}P;\n')
        f.write(f'ClarabelCscMatrix {prefix}A;\n')
        cone_str_list = []
        for cd in ['cone_dims_zero', 'cone_dims_nonneg']:
            if self.canon_constants[cd] > 0:
                cone_str_list.append(f'{self.canon_constants["cd_to_t"][cd]}({self.canon_constants[cd]})')
        cone_str_list.extend(['ClarabelExponentialConeT()'] * self.canon_constants['cone_dims_exp'])
        for cd in ['cone_dims_soc', 'cone_dims_psd', 'cone_dims_p3d']:
            for l in self.canon_constants[cd]:
                cone_str_list.append(f'{self.canon_constants["cd_to_t"][cd]}({l})')
        f.write(f'ClarabelSupportedConeT {prefix}cones[{self.canon_constants["n_cone_types"]}] = {{ {", ".join(cone_str_list)} }};\n')
        f.write(f'ClarabelDefaultSettings {prefix}settings;\n')
        f.write(f'ClarabelDefaultSolver *{prefix}solver = 0;\n')
        f.write(f'ClarabelDefaultSolution {prefix}solution;\n')
