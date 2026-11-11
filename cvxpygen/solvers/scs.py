"""
Copyright 2021-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
import shutil

import numpy as np
import scipy.sparse as spa

from cvxpygen.solvers import SolverInterface
from cvxpygen import utils
from cvxpygen.mappings import WorkspacePointerInfo, UpdatePendingLogic, \
    ParameterUpdateLogic, Setting
    

class SCSInterface(SolverInterface):
    solver_name = 'SCS'
    cvxpy_solver_name = 'SCS'
    solver_type = 'conic'
    supports_quad_obj = True
    canon_p_ids = ['P', 'c', 'd', 'A', 'b']
    canon_p_ids_constr_vec = ['b']
    parameter_update_structure = {
        'init': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic([], extra_condition = '!{prefix}Scs_Work', functions_if_false = ['PA']),
            function_call = '{prefix}Scs_Work = scs_init(&{prefix}Scs_D, &{prefix}Scs_K, &{prefix}Canon_Settings)',
        ),
        'PA': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['P', 'A'], '||', functions_if_false = ['bc']),
            function_call = '{prefix}Scs_Work = scs_init(&{prefix}Scs_D, &{prefix}Scs_K, &{prefix}Canon_Settings)',
        ),
        'bc': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['b', 'c'], '&&', ['b', 'c']),
            function_call = 'scs_update({prefix}Scs_Work, {prefix}Canon_Params.b, {prefix}Canon_Params.c)'
        ),
        'b': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['b']),
            function_call = 'scs_update({prefix}Scs_Work, {prefix}Canon_Params.b, SCS_NULL)'
        ),
        'c': ParameterUpdateLogic(
            update_pending_logic = UpdatePendingLogic(['c']),
            function_call = 'scs_update({prefix}Scs_Work, SCS_NULL, {prefix}Canon_Params.c)'
        )
    }
    solve_function_call = 'scs_solve({prefix}Scs_Work, &{prefix}Scs_Sol, &{prefix}Scs_Info, ({prefix}Scs_Work && {prefix}Canon_Settings.warm_start))'

    # header files
    header_files = ['"scs.h"']
    cmake_headers = [
        '${${PROJECT_NAME}_HDR}',
        '${DIRSRC}/private.h',
        '${${PROJECT_NAME}_LDL_EXTERNAL_HDR}',
        '${${PROJECT_NAME}_AMD_EXTERNAL_HDR}',
        ]
    cmake_sources = [
        '${${PROJECT_NAME}_SRC}',
        '${DIRSRC}/private.c',
        '${EXTERNAL}/qdldl/qdldl.c',
        '${${PROJECT_NAME}_AMD_EXTERNAL_SRC}'
        ]
    has_qdldl = True

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = False

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'Scs_Info.pobj',
        iterations = 'Scs_Info.iter',
        status = 'Scs_Info.status',
        primal_residual = 'Scs_Info.res_pri',
        dual_residual = 'Scs_Info.res_dual',
        primal_solution = 'scs_x',
        dual_solution = 'scs_{dual_var_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = True

    # solver status as integer vs. string
    status_is_int = False

    # float and integer types
    numeric_types = {'float': 'scs_float', 'int': 'scs_int'}

    # solver settings
    stgs_dynamically_allocated = False
    stgs_requires_extra_struct_type = False
    stgs_direct_write_ptr = None
    stgs_reset_function = {'name': 'scs_set_default_settings', 'ptr': None} # set 'ptr' to None if stgs not statically allocated in solver code
    stgs = {
        'normalize': Setting(type='cpg_int', default='1'),
        'scale': Setting(type='cpg_float', default='0.1'),
        'adaptive_scale': Setting(type='cpg_int', default='1'),
        'rho_x': Setting(type='cpg_float', default='1e-6'),
        'max_iters': Setting(type='cpg_int', default='1e5'),
        'eps_abs': Setting(type='cpg_float', default='1e-4'),
        'eps_rel': Setting(type='cpg_float', default='1e-4'),
        'eps_infeas': Setting(type='cpg_float', default='1e-7'),
        'alpha': Setting(type='cpg_float', default='1.5'),
        'time_limit_secs': Setting(type='cpg_float', default='0'),
        'verbose': Setting(type='cpg_int', default='0'),
        'warm_start': Setting(type='cpg_int', default='0'),
        'acceleration_lookback': Setting(type='cpg_int', default='0'),
        'acceleration_interval': Setting(type='cpg_int', default='1'),
        'write_data_filename': Setting(type='const char*', default='SCS_NULL'),
        'log_csv_filename': Setting(type='const char*', default='SCS_NULL')
    }
    
    # dual variables split into two vectors
    dual_var_split = False
    dual_var_names = ['y']

    # docu
    docu = 'https://www.cvxgrp.org/scs/api/c.html'

    def __init__(self, data, p_prob, enable_settings):
        n_var = p_prob.x.size
        n_eq = data['A'].shape[0]
        n_ineq = 0

        canon_constants = {'n': n_var, 'm': n_eq, 'z': p_prob.cone_dims.zero,
                           'l': p_prob.cone_dims.nonneg,
                           'q': np.array(p_prob.cone_dims.soc),
                           'qsize': len(p_prob.cone_dims.soc)}

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, p_prob, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0 or len(cone_dims.psd) > 0 or len(cone_dims.p3d) > 0:
            raise ValueError(
                'Code generation with SCS and exponential, positive semidefinite, or power cones '
                'is not supported yet.')

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      canon, gradient, prefix) -> None:

        # check if P is upper triangular
        parameter_canon = canon.parameter_canon
        if 'P' in parameter_canon.p and parameter_canon.p['P'] is not None:
            P_mat = parameter_canon.p['P']
            if spa.tril(P_mat, -1).data.size > 0:
                parameter_canon.p['P'] = spa.triu(P_mat, format='csc')

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'linsys', 'cmake']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'scs', dtc),
                            os.path.join(solver_code_dir, dtc))
        files_to_copy = ['scs.mk', 'CMakeLists.txt', 'LICENSE.txt']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'scs', fl),
                            os.path.join(solver_code_dir, fl))
            
        # modify scs_matrix.c
        utils.read_write_file(os.path.join(solver_code_dir, 'linsys', 'scs_matrix.c'),
                              lambda x: x.replace('#include "util.h"', '#include "util.h"\n#include "cones.h"'))

        # disable BLAS and LAPACK
        utils.read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'scs.mk'),
                              lambda x: x.replace('USE_LAPACK = 1', 'USE_LAPACK = 0'))

        # modify CMakeLists.txt
        cmake_replacements = [
            (' include/', ' ${CMAKE_CURRENT_SOURCE_DIR}/include/'),
            (' src/', ' ${CMAKE_CURRENT_SOURCE_DIR}/src/'),
            (' ${LINSYS}/', ' ${CMAKE_CURRENT_SOURCE_DIR}/${LINSYS}/')
        ]
        utils.read_write_file(os.path.join(solver_code_dir, 'CMakeLists.txt'),
                              lambda x: utils.multiple_replace(x, cmake_replacements))

    def cmake_context_extra(self) -> dict:
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        return {
            **super().cmake_context_extra(),
            'extra_cmake_include_dirs': [sdir + 'linsys'],
        }

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        matrices = ['P', 'A'] if parameter_canon.quad_obj else ['A']
        for m in matrices:
            f.write(f'\n// SCS matrix {m}\n')
            utils.write_struct_prot(f, f'{prefix}scs_{m}', 'ScsMatrix')
        f.write(f'\n// Struct containing SCS data\n')
        utils.write_struct_prot(f, f'{prefix}Scs_D', 'ScsData')
        if self.canon_constants['qsize'] > 0:
            f.write(f'\n// SCS array of SOC dimensions\n')
            utils.write_vec_prot(f, self.canon_constants['q'], f'{prefix}scs_q', 'cpg_int')
        f.write(f'\n// Struct containing SCS cone data\n')
        utils.write_struct_prot(f, f'{prefix}Scs_K', 'ScsCone')
        f.write(f'\n// Struct containing SCS settings\n')
        utils.write_struct_prot(f, f'{prefix}Canon_Settings', 'ScsSettings')
        f.write(f'\n// SCS solution\n')
        utils.write_vec_prot(f, np.zeros(self.canon_constants['n']), f'{prefix}scs_x', 'cpg_float')
        utils.write_vec_prot(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_y', 'cpg_float')
        utils.write_vec_prot(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_s', 'cpg_float')
        f.write(f'\n// Struct containing SCS solution\n')
        utils.write_struct_prot(f, f'{prefix}Scs_Sol', 'ScsSolution')
        f.write(f'\n// Struct containing SCS information\n')
        utils.write_struct_prot(f, f'{prefix}Scs_Info', 'ScsInfo')
        f.write(f'\n// Pointer to struct containing SCS workspace\n')
        utils.write_struct_prot(f, f'{prefix}Scs_Work', 'ScsWork*')


    def define_workspace(self, f, prefix, parameter_canon) -> None:
        matrices = ['P', 'A'] if parameter_canon.quad_obj else ['A']
        scs_PA_fields = ['x', 'i', 'p', 'm', 'n']
        scs_PA_casts = ['(cpg_float *) ', '(cpg_int *) ', '(cpg_int *) ', '', '']
        for m in matrices:
            f.write(f'\n// SCS matrix {m}\n')
            scs_PA_values = [f'&{prefix}canon_{m}_x', f'&{prefix}canon_{m}_i',
                            f'&{prefix}canon_{m}_p', str(self.canon_constants[('n' if m == 'P' else 'm')]),
                            str(self.canon_constants['n'])]
            utils.write_struct_def(f, scs_PA_fields, scs_PA_casts, scs_PA_values, f'{prefix}Scs_{m}', 'ScsMatrix')

        f.write(f'\n// Struct containing SCS data\n')
        scs_d_fields = ['m', 'n', 'A', 'P', 'b', 'c']
        scs_d_casts = ['', '', '', '', '(cpg_float *) ', '(cpg_float *) ']
        scs_d_values = [str(self.canon_constants['m']), str(self.canon_constants['n']),
                        f'&{prefix}Scs_A', (f'&{prefix}Scs_P' if parameter_canon.quad_obj else 'SCS_NULL'),
                        f'&{prefix}canon_b', f'&{prefix}canon_c']
        utils.write_struct_def(f, scs_d_fields, scs_d_casts, scs_d_values, f'{prefix}Scs_D', 'ScsData')

        if self.canon_constants['qsize'] > 0:
            f.write(f'\n// SCS array of SOC dimensions\n')
            utils.write_vec_def(f, self.canon_constants['q'], f'{prefix}scs_q', 'cpg_int')
            k_field_q_str = f'&{prefix}scs_q'
        else:
            k_field_q_str = 'SCS_NULL'

        f.write(f'\n// Struct containing SCS cone data\n')
        scs_k_fields = ['z', 'l', 'bu', 'bl', 'bsize', 'q', 'qsize', 's', 'ssize', 'ep', 'ed', 'p', 'psize']
        scs_k_casts = ['', '', '(cpg_float *) ', '(cpg_float *) ', '', '(cpg_int *) ', '', '(cpg_int *) ', '', '', '',
                    '(cpg_float *) ', '']
        scs_k_values = [str(self.canon_constants['z']), str(self.canon_constants['l']), 'SCS_NULL', 'SCS_NULL', '0',
                        k_field_q_str, str(self.canon_constants['qsize']), 'SCS_NULL', '0', '0', '0', 'SCS_NULL', '0']
        utils.write_struct_def(f, scs_k_fields, scs_k_casts, scs_k_values, f'{prefix}Scs_K', 'ScsCone')

        f.write(f'\n// Struct containing SCS settings\n')
        scs_stgs_fields = list(self.stgs_names_to_default.keys())
        scs_stgs_casts = [''] * len(scs_stgs_fields)
        scs_stgs_values = list(self.stgs_names_to_default.values())
        utils.write_struct_def(f, scs_stgs_fields, scs_stgs_casts, scs_stgs_values, f'{prefix}Canon_Settings', 'ScsSettings')

        f.write(f'\n// SCS solution\n')
        utils.write_vec_def(f, np.zeros(self.canon_constants['n']), f'{prefix}scs_x', 'cpg_float')
        utils.write_vec_def(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_y', 'cpg_float')
        utils.write_vec_def(f, np.zeros(self.canon_constants['m']), f'{prefix}scs_s', 'cpg_float')

        f.write(f'\n// Struct containing SCS solution\n')
        scs_sol_fields = ['x', 'y', 's']
        scs_sol_casts = ['(cpg_float *) ', '(cpg_float *) ', '(cpg_float *) ']
        scs_sol_values = [f'&{prefix}scs_x', f'&{prefix}scs_y', f'&{prefix}scs_s']
        utils.write_struct_def(f, scs_sol_fields, scs_sol_casts, scs_sol_values, f'{prefix}Scs_Sol', 'ScsSolution')

        f.write(f'\n// Struct containing SCS information\n')
        scs_info_fields = ['iter', 'status', 'status_val', 'scale_updates', 'pobj', 'dobj', 'res_pri', 'res_dual',
                        'gap', 'res_infeas', 'res_unbdd_a', 'res_unbdd_p', 'comp_slack', 'setup_time', 'solve_time',
                        'scale', 'rejected_accel_steps', 'accepted_accel_steps', 'lin_sys_time', 'cone_time',
                        'accel_time']
        scs_info_casts = [''] * len(scs_info_fields)
        scs_info_values = ['0', '"unknown"', '0', '0', '0', '0', '99', '99', '99', '99', '99', '99', '99', '0', '0',
                        '1', '0', '0', '0', '0', '0']
        utils.write_struct_def(f, scs_info_fields, scs_info_casts, scs_info_values, f'{prefix}Scs_Info', 'ScsInfo')

        f.write(f'\n// Pointer to struct containing SCS workspace\n')
        f.write(f'ScsWork* {prefix}Scs_Work = 0;\n')
