"""
Copyright 2022-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
import shutil

import numpy as np

from cvxpygen import utils
from cvxpygen.solvers import SolverInterface
from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, \
    WorkspacePointerInfo, UpdatePendingLogic, ParameterUpdateLogic, Setting
    

class ECOSInterface(SolverInterface):
    solver_name = 'ECOS'
    cvxpy_solver_name = 'ECOS'
    solver_type = 'conic'
    canon_p_ids = ['c', 'd', 'A', 'b', 'G', 'h']
    canon_p_ids_constr_vec = ['b', 'h']
    solve_function_call = '{prefix}ecos_flag = ECOS_solve({prefix}ecos_workspace)'

    # header files
    header_files = ['"ecos.h"']
    cmake_headers = ['${ecos_headers}']
    cmake_sources = ['${ecos_sources}']
    has_qdldl = False

    # preconditioning of problem data happening in-memory
    inmemory_preconditioning = True

    # workspace
    ws_statically_allocated_in_solver_code = False
    ws_ptrs = WorkspacePointerInfo(
        objective_value = 'ecos_workspace->info->pcost',
        iterations = 'ecos_workspace->info->iter',
        status = 'ecos_flag',
        primal_residual = 'ecos_workspace->info->pres',
        dual_residual = 'ecos_workspace->info->dres',
        primal_solution = 'ecos_workspace->x',
        dual_solution = 'ecos_workspace->{dual_var_name}',
        settings = 'ecos_workspace->stgs->{setting_name}'
    )

    # solution vectors statically allocated
    sol_statically_allocated = False

    # solver status as integer vs. string
    status_is_int = True

    # float and integer types
    numeric_types = {'float': 'double', 'int': 'int'}

    # solver settings
    stgs_dynamically_allocated = True
    stgs_requires_extra_struct_type = True
    stgs_direct_write_ptr = None
    stgs_reset_function = None
    stgs = {
        'feastol': Setting(type='cpg_float', default='1e-8'),
        'abstol': Setting(type='cpg_float', default='1e-8'),
        'reltol': Setting(type='cpg_float', default='1e-8'),
        'feastol_inacc': Setting(type='cpg_float', default='1e-4'),
        'abstol_inacc': Setting(type='cpg_float', default='5e-5'),
        'reltol_inacc': Setting(type='cpg_float', default='5e-5'),
        'maxit': Setting(type='cpg_int', default='100', name_cvxpy='max_iters')
    }

    # dual variables split into y and z vectors
    dual_var_split = True
    dual_var_names = ['y', 'z']

    # docu
    docu = 'https://github.com/embotech/ecos/wiki/Usage-from-C'

    def __init__(self, canon_maps, enable_settings):
        cone_dims = canon_maps.cone_dims
        n_var = canon_maps.maps['c'][1].size
        n_eq = cone_dims.zero
        n_ineq = canon_maps.structures['G'].shape[0]

        canon_constants = {'n': n_var, 'm': n_ineq, 'p': n_eq,
                           'l': cone_dims.nonneg,
                           'n_cones': len(cone_dims.soc),
                           'q': np.array(cone_dims.soc),
                           'e': cone_dims.exp}

        self.parameter_update_structure = {
            'init': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic([], extra_condition='!{prefix}ecos_workspace', functions_if_false=['AbcGh']),
                function_call=f'{{prefix}}cpg_copy_all();\n'
                            f'    {{prefix}}ecos_workspace = ECOS_setup({canon_constants["n"]}, {canon_constants["m"]}, {canon_constants["p"]}, {canon_constants["l"]}, {canon_constants["n_cones"]}'
                            f', {"0" if canon_constants["n_cones"] == 0 else "(int *) &{prefix}ecos_q"}, {canon_constants["e"]}'
                            f', {{prefix}}Canon_Params_conditioning.G->x, {{prefix}}Canon_Params_conditioning.G->p, {{prefix}}Canon_Params_conditioning.G->i'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->x"}'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->p"}'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->i"}'
                            f', {{prefix}}Canon_Params_conditioning.c, {{prefix}}Canon_Params_conditioning.h'
                            f', {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.b"})'
            ),
            'AbcGh': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['A', 'b', 'G'], '||', ['c', 'h']),
                function_call=f'{{prefix}}cpg_copy_all();\n'
                            f'      ECOS_updateData({{prefix}}ecos_workspace, {{prefix}}Canon_Params_conditioning.G->x, {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.A->x"}'
                            f', {{prefix}}Canon_Params_conditioning.c, {{prefix}}Canon_Params_conditioning.h, {"0" if canon_constants["p"] == 0 else "{prefix}Canon_Params_conditioning.b"})'
            ),
            'c': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['c']),
                function_call=f'{{prefix}}cpg_copy_c();\n'
                            f'        for (i=0; i<{canon_constants["n"]}; i++) {{{{ ecos_updateDataEntry_c({{prefix}}ecos_workspace, i, {{prefix}}Canon_Params_conditioning.c[i]); }}}}'
            ),
            'h': ParameterUpdateLogic(
                update_pending_logic=UpdatePendingLogic(['h']),
                function_call=f'{{prefix}}cpg_copy_h();\n'
                            f'        for (i=0; i<{canon_constants["m"]}; i++) {{{{ ecos_updateDataEntry_h({{prefix}}ecos_workspace, i, {{prefix}}Canon_Params_conditioning.h[i]); }}}}'
            )
        }

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, canon_constants, enable_settings)

    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        if cone_dims.exp > 0:
            raise ValueError(
                'Code generation with ECOS and exponential cones is not supported yet.')

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return True

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return True

    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      canon, gradient, prefix) -> None:

        # copy sources
        if os.path.isdir(solver_code_dir):
            shutil.rmtree(solver_code_dir)
        os.mkdir(solver_code_dir)
        dirs_to_copy = ['src', 'include', 'external', 'ecos_bb']
        for dtc in dirs_to_copy:
            shutil.copytree(os.path.join(cvxpygen_directory, 'solvers', 'ecos', dtc),
                            os.path.join(solver_code_dir, dtc))
        
        files_to_copy = ['CMakeLists.txt', 'COPYING']
        for fl in files_to_copy:
            shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', fl),
                            os.path.join(solver_code_dir, fl))
        
        shutil.copyfile(os.path.join(cvxpygen_directory, 'solvers', 'ecos', 'COPYING'),
                        os.path.join(code_dir, 'COPYING'))

        # adjust print level
        utils.read_write_file(os.path.join(code_dir, 'c', 'solver_code', 'include', 'glblopts.h'),
                              lambda x: x.replace('#define PRINTLEVEL (2)', '#define PRINTLEVEL (0)'))

    def cmake_context_extra(self) -> dict:
        sdir = '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/'
        return {
            **super().cmake_context_extra(),
            'extra_cmake_include_dirs': [
                sdir + 'external/SuiteSparse_config',
                sdir + 'external/amd/include',
                sdir + 'external/ldl/include',
            ],
        }

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            utils.write_vec_prot(f, self.canon_constants['q'], f'{prefix}ecos_q', 'cpg_int')
        f.write('\n// ECOS workspace\n')
        f.write(f'extern pwork* {prefix}ecos_workspace;\n')
        f.write('\n// ECOS exit flag\n')
        f.write(f'extern cpg_int {prefix}ecos_flag;\n')

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        if self.canon_constants['n_cones'] > 0:
            f.write('\n// ECOS array of SOC dimensions\n')
            utils.write_vec_def(f, self.canon_constants['q'], f'{prefix}ecos_q', 'cpg_int')
        f.write('\n// ECOS workspace\n')
        f.write(f'pwork* {prefix}ecos_workspace = 0;\n')
        f.write('\n// ECOS exit flag\n')
        f.write(f'cpg_int {prefix}ecos_flag = -99;\n')
