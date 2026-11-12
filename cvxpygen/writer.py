"""
Copyright 2021-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
from datetime import datetime
from typing import Optional

import numpy as np
import scipy.sparse as sp
import cvxpy as cp

from cvxpygen import utils
from cvxpygen.mappings import Configuration, Canon


class CCodeWriter:
    """
    Writes all C/C++ source files for a canonicalized CVXPY problem.

    Delegates to the low-level helper functions in utils.py; this class
    exists solely to give those calls a coherent owner and to group related
    writes into clearly-named methods.
    """

    def __init__(
        self,
        configuration: Configuration,
        canon: Canon,
        solver_interface,
        gradient_interface=None,
        canon_solver: Optional[Canon] = None,
        canon_gradient: Optional[Canon] = None,
    ) -> None:
        self.configuration = configuration
        self.canon = canon
        self.solver_interface = solver_interface
        self.gradient_interface = gradient_interface
        self.canon_solver = canon_solver
        self.canon_gradient = canon_gradient

        # Convenience references
        self._pvi = canon.prim_variable_info
        self._dvi = canon.dual_variable_info
        self._pi = canon.parameter_info
        self._pc = canon.parameter_canon

        # Directory shortcuts
        c_dir = os.path.join(configuration.code_dir, 'c')
        self._c_dir = c_dir
        self._include_dir = os.path.join(c_dir, 'include')
        self._src_dir = os.path.join(c_dir, 'src')
        self._solver_code_dir = os.path.join(c_dir, 'solver_code')
        self._osqp_code_dir = os.path.join(c_dir, 'osqp_code')
        self._cpp_dir = os.path.join(configuration.code_dir, 'cpp')

    # ── public entry point ────────────────────────────────────────────────────

    def write(self) -> None:
        """Write all generated files."""
        self._write_top_level_files()
        self._write_workspace()
        self._write_solve()
        if self.configuration.gradient:
            self._write_gradient()
        self._write_example()
        self._write_python_module()
        self._write_python_solver()
        self._update_cmake()
        self._update_readme()

    # ── private writer methods ────────────────────────────────────────────────
    
    def _write_top_level_files(self) -> None:
        si = self.solver_interface
        utils.render_template_to_file(
            'LICENSE.jinja2', self.configuration.code_dir
        )
        utils.render_template_to_file(
            '__init__.py.jinja2', self.configuration.code_dir
        )
        cmake_ctx = {**utils.cmake_context(self.configuration), **si.cmake_context_extra()}
        if self.configuration.gradient_two_stage:
            # Add OSQP include dirs AFTER solver include dirs to prevent OSQP headers
            # (e.g., osqp_code/inc/private/util.h) from shadowing same-named solver
            # headers (e.g., solver_code/include/util.h for SCS).
            cmake_ctx['extra_cmake_include_dirs'] = cmake_ctx.get('extra_cmake_include_dirs', []) + [
                '${CMAKE_CURRENT_SOURCE_DIR}/osqp_code/',
                '${CMAKE_CURRENT_SOURCE_DIR}/osqp_code/inc/private',
                '${CMAKE_CURRENT_SOURCE_DIR}/osqp_code/inc/public',
            ]
        utils.render_template_to_file(
            'CMakeLists.txt.jinja2', self._c_dir,
            cmake_ctx
        )
        utils.render_template_to_file(
            'pyproject.toml.jinja2', self.configuration.code_dir,
            {'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S")}
        )
        utils.render_template_to_file(
            'root-CMakeLists.txt.jinja2', self.configuration.code_dir,
            cmake_ctx
        )

    def _write_workspace(self) -> None:
        cfg = self.configuration
        si = self.solver_interface

        # For two-stage gradient, temporarily redirect workspace pointers so
        # that the main workspace points at the intermediate OSQP solution.
        if cfg.gradient_two_stage:
            primal_ptr = si.ws_ptrs.primal_solution
            dual_ptr = si.ws_ptrs.dual_solution
            if si.ret_prim_func_exists(self.canon_solver.prim_variable_info):
                si.ws_ptrs.primal_solution = f'gradient_{cfg.prefix}sol_x'
            si.ws_ptrs.dual_solution = f'gradient_{cfg.prefix}sol_y'

        utils.write_file(
            os.path.join(self._include_dir, 'cpg_workspace.h'), 'w',
            utils.write_workspace_prot,
            cfg, self._pvi, self._dvi, self._pi, self._pc, si, True,
        )
        utils.write_file(
            os.path.join(self._src_dir, 'cpg_workspace.c'), 'w',
            utils.write_workspace_def,
            cfg, self._pvi, self._dvi, self._pi, self._pc, si, True,
        )

        if cfg.gradient_two_stage:
            # Restore original pointers for the rest of code generation.
            si.ws_ptrs.primal_solution = primal_ptr
            si.ws_ptrs.dual_solution = dual_ptr

    def _write_solve(self) -> None:
        cfg = self.configuration
        si = self.solver_interface

        # For two-stage gradient, temporarily redirect workspace pointers so
        # that the main workspace points at the intermediate OSQP solution.
        if cfg.gradient_two_stage:
            pc_gradient = self.canon_gradient.parameter_canon
        else:
            pc_gradient = None

        utils.write_file(
            os.path.join(self._include_dir, 'cpg_solve.h'), 'w',
            utils.write_solve_prot,
            cfg, self._pvi, self._dvi, self._pi, self._pc, si, pc_gradient,
        )
        utils.write_file(
            os.path.join(self._src_dir, 'cpg_solve.c'), 'w',
            utils.write_solve_def,
            cfg, self._pvi, self._dvi, self._pi, self._pc, si, pc_gradient,
        )
    
    def _write_gradient_def(self, f, configuration,
                           variable_info_first, dual_variable_info_first,
                           variable_info_second, dual_variable_info_second,
                           parameter_info, parameter_canon, solver_interface,
                           gradient_interface):
        """
        Write parameter initialization function to file
        """
        
        if configuration.gradient_two_stage:
            prefix = f'gradient_{configuration.prefix}'
        else:
            prefix = configuration.prefix
        # CPG_OSQP_Grad is always written with configuration.prefix (not gradient_prefix)
        osqp_prefix = configuration.prefix

        n = gradient_interface.n_var
        nl = gradient_interface.n_eq
        nu = gradient_interface.n_eq + gradient_interface.n_ineq
        N = n + nu
        NP = len(parameter_info.flat_usp)

        utils.write_description(f, 'c', 'Function definitions')
        f.write('#include "cpg_gradient.h"\n')
        if configuration.gradient_two_stage:
            f.write('#include "cpg_gradient_workspace.h"\n')
            f.write('#include "cpg_workspace.h"\n')
        f.write('#include "cpg_osqp_grad_workspace.h"\n')
        f.write('#include "cpg_osqp_grad_compute.h"\n')
        f.write('\n')

        f.write('static cpg_int i, j, k;\n\n')
        
        if configuration.gradient_two_stage:
            
            ret_prim_func_needed = solver_interface.ret_prim_func_exists(variable_info_second)
            
            # define sol_x and sol_y to point to the dual solution
            result_prefix = configuration.prefix if not solver_interface.ws_statically_allocated_in_solver_code else ''
            if ret_prim_func_needed:
                utils.write_vec_def(f, np.zeros(n), f'{prefix}sol_x', 'cpg_float')
            else:
                f.write(f'cpg_float* {prefix}sol_x = (cpg_float*) &{result_prefix}{solver_interface.ws_ptrs.primal_solution} + {variable_info_second.name_to_offset["osqp_x"]};\n')
            utils.write_vec_def(f, np.zeros(nu), f'{prefix}sol_y', 'cpg_float')
            
            # retrieve intermediate primal
            if ret_prim_func_needed:
                offset = variable_info_second.name_to_offset['osqp_x']
                x_term = f'{result_prefix}{solver_interface.ws_ptrs.primal_solution}[{offset} + i]'
                f.write(f'\nvoid {prefix}cpg_retrieve_intermediate_primal(){{\n')
                f.write(f'  for(i=0; i<{n}; i++){{\n')
                f.write(f'    {prefix}sol_x[i] = {x_term};\n')
                f.write('  }\n')
                f.write('}\n\n')
            
            # retrieve intermediate dual
            vec_l = dual_variable_info_second.name_to_vec['d0']
            vec_u = dual_variable_info_second.name_to_vec['d1']
            offset_l = dual_variable_info_second.name_to_offset['d0']
            offset_u = dual_variable_info_second.name_to_offset['d1']
            size_l = dual_variable_info_second.name_to_size['d0']
            size_u = dual_variable_info_second.name_to_size['d1']
            l_term = f'{result_prefix}{solver_interface.ws_ptrs.dual_solution.format(dual_var_name=vec_l)}[{offset_l} + i]'
            u_term = f'{result_prefix}{solver_interface.ws_ptrs.dual_solution.format(dual_var_name=vec_u)}[{offset_u} + i]'
            f.write(f'\nvoid {prefix}cpg_retrieve_intermediate_dual(){{\n')
            f.write(f'  for(i=0; i<{size_l}; i++){{\n')
            f.write(f'    {prefix}sol_y[i] = {u_term} - {l_term};\n')
            f.write('  }\n')
            f.write(f'  for(i={size_l}; i<{size_u}; i++){{\n')
            f.write(f'    {prefix}sol_y[i] = {u_term};\n')
            f.write('  }\n')
            f.write('}\n\n')
            
            for p_id, mapping in parameter_canon.p_id_to_mapping.items():
                if parameter_canon.p_id_to_changes[p_id]:
                    f.write(f'void {prefix}cpg_canonicalize_{p_id}(){{\n')
                    s = '->x' if p_id.isupper() else ''
                    utils.write_canonicalize(f, p_id, s, mapping, prefix, configuration.prefix)
                    f.write('}\n\n')
        
        f.write('// Update user-defined variable deltas\n')
        for name, size in variable_info_first.name_to_size.items():
            if size == 1:
                f.write(f'void {configuration.prefix}cpg_update_d{name}(cpg_float val){{\n')
                f.write(f'  {osqp_prefix}CPG_OSQP_Grad.dx[{variable_info_first.name_to_offset[name]}] = val;\n')
            else:
                f.write(f'void {configuration.prefix}cpg_update_d{name}(cpg_int idx, cpg_float val){{\n')
                f.write(f'  {osqp_prefix}CPG_OSQP_Grad.dx[{variable_info_first.name_to_offset[name]}+idx] = val;\n')
            f.write('}\n')

        f.write('\n// End-to-end gradient\n')
        f.write(f'void {configuration.prefix}cpg_gradient(){{\n')
        if configuration.gradient_two_stage:
            for p_id, changes in parameter_canon.p_id_to_changes.items():
                if changes:
                    f.write(f'  if ({prefix}Canon_Outdated_Grad.{p_id}) {{\n')
                    f.write(f'    {prefix}cpg_canonicalize_{p_id}();\n')
                    f.write('  }\n')
        changing_matrices = []
        for p_id, changes in parameter_canon.p_id_to_changes.items():
            if p_id.isupper() and changes:
                changing_matrices.append(p_id)
                f.write(f'  if ({prefix}Canon_Outdated_Grad.{p_id}) {{\n')
                f.write(f'    cpg_{p_id}_to_K((cpg_grad_csc*) {prefix}Canon_Params.{p_id}, {osqp_prefix}CPG_OSQP_Grad.K, {osqp_prefix}CPG_OSQP_Grad.K_true);\n')
                f.write('  }\n')
        f.write(f'  if ({osqp_prefix}CPG_OSQP_Grad.init) {{\n')
        f.write('    cpg_ldl_symbolic();\n')
        f.write('    cpg_ldl_numeric();\n')
        f.write(f'    for (j=0; j<{N-1}; j++){{\n')
        f.write(f'      for (k={osqp_prefix}CPG_OSQP_Grad.L->p[j]; k<{osqp_prefix}CPG_OSQP_Grad.L->p[j+1]; k++){{\n')
        f.write(f'        i = {osqp_prefix}CPG_OSQP_Grad.L->i[k];\n')
        f.write(f'        {osqp_prefix}CPG_OSQP_Grad.Lmask[({2*N-3}-j)*j/2+i-1] = 2;\n')
        f.write('      }\n')
        f.write('    }\n')
        f.write(f'    {osqp_prefix}CPG_OSQP_Grad.init = 0;\n')
        # If any of {prefix}Canon_Outdated_Grad.{p_id} with p_id in changing_matrices is true, then call cpg_ldl_numeric()
        if len(changing_matrices) > 0:
            f.write('  } else {\n')
            f.write(f'    if ({f" || ".join([f"{prefix}Canon_Outdated_Grad.{p_id}" for p_id in changing_matrices])}) {{\n')
            f.write(f'      cpg_ldl_numeric();\n')
            # set all CPG_OSQP_Grad.a[i] to 1
            f.write(f'      for(i=0; i<{N - n - nl}; i++){{\n')
            f.write(f'        {osqp_prefix}CPG_OSQP_Grad.a[i] = 1;\n')
            f.write('      }\n')
            f.write('    }\n')
        f.write('  }\n')
        f.write('  // Canonical gradient\n')
        f.write(f'  cpg_osqp_gradient();\n')
        f.write('  // Un-canonicalize\n')
        f.write(f'  for(j=0; j<{NP}; j++){{\n')
        f.write(f'      {configuration.prefix}cpg_dp[j] = 0.0;\n')
        f.write('  }\n')
        if parameter_canon.p_id_to_changes['q']:
            f.write(f'  for(j=0; j<{n}; j++){{\n')
            f.write(f'      for(k={prefix}canon_q_map.p[j]; k<{prefix}canon_q_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_q_map.i[k]] += {prefix}canon_q_map.x[k]*{osqp_prefix}CPG_OSQP_Grad.dq[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['l']:
            f.write(f'  for(j=0; j<{nl}; j++){{\n')
            f.write(f'      for(k={prefix}canon_l_map.p[j]; k<{prefix}canon_l_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_l_map.i[k]] += {prefix}canon_l_map.x[k]*{osqp_prefix}CPG_OSQP_Grad.dl[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['u']:
            f.write(f'  for(j=0; j<{nu}; j++){{\n')
            f.write(f'      for(k={prefix}canon_u_map.p[j]; k<{prefix}canon_u_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_u_map.i[k]] += {prefix}canon_u_map.x[k]*{osqp_prefix}CPG_OSQP_Grad.du[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['P']:
            f.write(f'  for(j=0; j<{parameter_canon.p["P"].nnz}; j++){{\n')
            f.write(f'      for(k={prefix}canon_P_map.p[j]; k<{prefix}canon_P_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_P_map.i[k]] += {prefix}canon_P_map.x[k]*{osqp_prefix}CPG_OSQP_Grad.dP->x[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        if parameter_canon.p_id_to_changes['A']:
            f.write(f'  for(j=0; j<{parameter_canon.p["A"].nnz}; j++){{\n')
            f.write(f'      for(k={prefix}canon_A_map.p[j]; k<{prefix}canon_A_map.p[j+1]; k++){{\n')
            f.write(f'          {configuration.prefix}cpg_dp[{prefix}canon_A_map.i[k]] += {prefix}canon_A_map.x[k]*{osqp_prefix}CPG_OSQP_Grad.dA->x[j];\n')
            f.write('      }\n')
            f.write('  }\n')
        f.write('  // Reset dx\n')
        f.write(f'  for(i=0; i<{n}; i++){{\n')
        f.write(f'      {osqp_prefix}CPG_OSQP_Grad.dx[i] = 0.0;\n')
        f.write('  }\n')
        f.write('  // Reset flags for outdated canonical parameters\n')
        for p_id, changes in parameter_canon.p_id_to_changes.items():
            if changes:
                f.write(f'  {prefix}Canon_Outdated_Grad.{p_id} = 0;\n')
        f.write('}\n\n')
            
            
    def _write_gradient_prot(self, f, configuration,
                            variable_info_first, dual_variable_info_first,
                            variable_info_second, dual_variable_info_second,
                            parameter_info, parameter_canon, solver_interface,
                            gradient_interface):
        """
        Write function declarations to file
        """
        
        if configuration.gradient_two_stage:
            prefix = f'gradient_{configuration.prefix}'
        else:
            prefix = configuration.prefix

        utils.write_description(f, 'c', 'Function declarations')
        f.write('#include "cpg_workspace.h"\n')
        #f.write('#include "osqp_api_types.h"\n')
        
        if configuration.gradient_two_stage:
            ret_prim_func_needed = solver_interface.ret_prim_func_exists(variable_info_second)
            if ret_prim_func_needed:
                f.write(f'\nextern void {prefix}cpg_retrieve_intermediate_primal();\n')
            f.write(f'\nextern void {prefix}cpg_retrieve_intermediate_dual();\n')
            if ret_prim_func_needed:
                utils.write_vec_prot(f, np.zeros(gradient_interface.n_var), f'{prefix}sol_x', 'cpg_float')
            else:
                f.write(f'\nextern cpg_float* {prefix}sol_x;\n')
            utils.write_vec_prot(f, np.zeros(gradient_interface.n_eq + gradient_interface.n_ineq), f'{prefix}sol_y', 'cpg_float')
        
        f.write('\n// Un-retrieve\n')
        for name, size in variable_info_first.name_to_size.items():
            if size == 1:
                f.write(f'extern void {configuration.prefix}cpg_update_d{name}(cpg_float val);\n')
            else:
                f.write(f'extern void {configuration.prefix}cpg_update_d{name}(cpg_int idx, cpg_float val);\n')
        
        f.write('\n// End-to-end gradient\n')
        f.write(f'extern void {configuration.prefix}cpg_gradient();\n')
        
        
    def _write_gradient_workspace_def(self, f, prefix, parameter_canon):
        """
        Write canonical gradient workspace to file
        """
        
        n = parameter_canon.p['P'].shape[0]
        N = n + parameter_canon.p['A'].shape[0]
        nl = self.gradient_interface.n_eq
        n_ineq = N - n - nl
        
        P_upper = sp.triu(parameter_canon.p['P'], format='csc')
        K = sp.bmat([
            [P_upper + 1e-6 * sp.eye(n), parameter_canon.p['A'].T],
            [None, - 1e-6 * sp.eye(N-n)]
        ], format='csc')
        
        K_true = sp.bmat([
            [utils.symmetric_from_upper(parameter_canon.p['P']), parameter_canon.p['A'].T],
            [parameter_canon.p['A'], None]
        ], format='csr')
        
        utils.write_description(f, 'c', 'Static workspace allocation for canonical gradient computation')
        f.write('#include "cpg_osqp_grad_workspace.h"\n\n')
        
        workspace = [
            ('a',       'int',      np.ones(N-n-nl, dtype=int)),
            ('etree',   'int',      np.zeros(N, dtype=int)),
            ('Lnz',     'int',      np.zeros(N, dtype=int)),
            ('iwork',   'int',      np.zeros(3*N, dtype=int)),
            ('bwork',   'int',      np.zeros(N, dtype=int)),
            ('fwork',   'float',    np.zeros(N)),
            ('L',       'csc_L',    N),
            ('Lmask',   'int',      np.zeros((N-1)*N//2, dtype=int)),
            ('Lcount',  'int',      np.zeros((N-1)*N//2, dtype=int)),
            ('D',       'float',    np.ones(N)),
            ('Dinv',    'float',    np.ones(N)),
            ('K',       'csc',      K),
            ('K_true',  'csc',      K_true),
            ('rhs',     'float',    np.zeros(N)),
            ('delta',   'float',    np.zeros(N)),
            ('c',       'float',    np.zeros(N)),
            ('w',       'float',    np.zeros(N)),
            ('wi',      'int',      np.arange(N)),
            ('widel',   'int',      np.zeros(n_ineq * n_ineq, dtype=int)),
            ('widel_sz','int',      np.zeros(n_ineq, dtype=int)),
            ('l',       'float',    np.zeros(N)),
            ('li',      'int',      np.arange(N)),
            ('lx',      'float',    np.zeros(N)),
            ('dx',      'float',    np.zeros(n)),
            ('r',       'float',    np.zeros(N)),
            ('dq',      'float',    np.zeros(n)),
            ('dl',      'float',    np.zeros(N-n)),
            ('du',      'float',    np.zeros(N-n)),
            ('dP',      'csc',      0*parameter_canon.p['P']),
            ('dA',      'csc',      0*parameter_canon.p['A'])
        ]
        
        for name, typ, value in workspace:
            if typ == 'csc':
                utils.write_mat_def(f, value, f'{prefix}cpg_osqp_grad_{name}', qualifier='grad')
            elif typ == 'csc_L':
                utils.write_L_def(f, value, f'{prefix}cpg_osqp_grad_{name}', qualifier='grad')
            else:
                utils.write_vec_def(f, value, f'{prefix}cpg_osqp_grad_{name}', 'cpg_' + typ, qualifier='grad')
                    
        OSQP_Grad_fiels = ['init'] + [w[0] for w in workspace]
        OSQP_Grad_casts = [''] + [f'{utils.type_to_cast(w[1], qualifier="grad")}&' for w in workspace]
        OSQP_Grad_values = ['1'] + [f'{prefix}cpg_osqp_grad_{v}' for v in OSQP_Grad_fiels[1:]]
        utils.write_struct_def(f, OSQP_Grad_fiels, OSQP_Grad_casts, OSQP_Grad_values, f'{prefix}CPG_OSQP_Grad', 'CPG_OSQP_Grad_t')        

    def _write_gradient_explicit(self) -> None:
        """Write gradient C files for explicit mode."""
        cfg = self.configuration
        si = self.solver_interface
        pc = self._pc
        pvi = self._pvi
        pi = self._pi

        n_sol = sum(pvi.name_to_size_reduced.values())
        n_params = pc.n_param_reduced   # = PDAQP_N_PARAMETER
        NP = len(pi.flat_usp)
        prefix = cfg.prefix

        # ── cpg_gradient.h ──────────────────────────────────────────────────
        def write_prot(f):
            utils.write_description(f, 'c', 'Function declarations (explicit gradient)')
            f.write('#include "cpg_workspace.h"\n')
            f.write('\n// Update user-defined variable deltas\n')
            for name, size in pvi.name_to_size.items():
                if size == 1:
                    f.write(f'extern void {prefix}cpg_update_d{name}(cpg_float val);\n')
                else:
                    f.write(f'extern void {prefix}cpg_update_d{name}(cpg_int idx, cpg_float val);\n')
            f.write('\n// End-to-end gradient\n')
            f.write(f'extern void {prefix}cpg_gradient();\n')

        utils.write_file(
            os.path.join(self._include_dir, 'cpg_gradient.h'), 'w', write_prot,
        )

        # ── cpg_gradient.c ──────────────────────────────────────────────────
        def write_def(f):
            utils.write_description(f, 'c', 'Function definitions (explicit gradient)')
            f.write('#include "cpg_gradient.h"\n')
            f.write('#include "cpg_workspace.h"\n')
            f.write('#include "../solver_code/include/pdaqp.h"\n\n')
            f.write('static cpg_int i, j, kk;\n')
            f.write(f'static cpg_float cpg_dx[{n_sol}];\n')
            f.write(f'static cpg_float cpg_dtheta[{n_params}];\n\n')

            # cpg_update_d<name> functions
            f.write('// Update user-defined variable deltas\n')
            for name, size in pvi.name_to_size.items():
                offset = pvi.name_to_offset[name]
                if size == 1:
                    f.write(f'void {prefix}cpg_update_d{name}(cpg_float val){{\n')
                    f.write(f'  cpg_dx[{offset}] = val;\n')
                else:  
                    f.write(f'void {prefix}cpg_update_d{name}(cpg_int idx, cpg_float val){{\n')
                    f.write(f'  cpg_dx[{offset}+idx] = val;\n')
                f.write('}\n')

            # cpg_gradient()
            f.write('\n// End-to-end gradient\n')
            f.write(f'void {prefix}cpg_gradient(){{\n')

            # Canonical diff dtheta = Z_active[:, :n_params].T @ dx
            f.write('  // Step 1: dtheta = Z_active[:, :n_params].T @ dx\n')
            f.write('  int base = pdaqp_hp_list[pdaqp_active_region] * (PDAQP_N_PARAMETER+1) * PDAQP_N_SOLUTION;\n')
            f.write(f'  for(j=0; j<{n_params}; j++){{\n')
            f.write(f'    cpg_dtheta[j] = 0.0;\n')
            f.write(f'    for(i=0; i<{n_sol}; i++){{\n')
            f.write(f'      cpg_dtheta[j] += (cpg_float)pdaqp_feedbacks[base + i*(PDAQP_N_PARAMETER+1) + j] * cpg_dx[i];\n')
            f.write('    }\n')
            f.write('  }\n\n')

            #  Un-canonicalize dtheta to dp via canon q/u maps and th_mask
            f.write('  // Un-canonicalize\n')
            f.write(f'  for(j=0; j<{NP}; j++) {prefix}cpg_dp[j] = 0.0;\n')
            n_var = si.n_var
            k = 0
            for i, active in enumerate(pc.th_mask):
                if not active:
                    continue
                if i < n_var:
                    if pc.p_id_to_changes.get('q', False):
                        f.write(f'  for(kk={prefix}canon_q_map.p[{i}];'
                                f' kk<{prefix}canon_q_map.p[{i+1}]; kk++){{\n')
                        f.write(f'    {prefix}cpg_dp[{prefix}canon_q_map.i[kk]]'
                                f' += {prefix}canon_q_map.x[kk] * cpg_dtheta[{k}];\n')
                        f.write('  }\n')
                else:
                    i_u = i - n_var
                    if pc.p_id_to_changes.get('u', False):
                        f.write(f'  for(kk={prefix}canon_u_map.p[{i_u}];'
                                f' kk<{prefix}canon_u_map.p[{i_u+1}]; kk++){{\n')
                        f.write(f'    {prefix}cpg_dp[{prefix}canon_u_map.i[kk]]'
                                f' += {prefix}canon_u_map.x[kk] * cpg_dtheta[{k}];\n')
                        f.write('  }\n')
                k += 1

            # reset dx
            f.write(f'\n  // Reset dx\n')
            f.write(f'  for(i=0; i<{n_sol}; i++) cpg_dx[i] = 0.0;\n')
            f.write('}\n\n')

        utils.write_file(
            os.path.join(self._src_dir, 'cpg_gradient.c'), 'w', write_def,
        )

    def _write_gradient(self) -> None:

        cfg = self.configuration

        if cfg.explicit:
            self._write_gradient_explicit()
            return

        si = self.solver_interface
        gi = self.gradient_interface

        utils.render_template_to_file(
            'cpg_osqp_grad_compute.c.jinja2', self._src_dir,
            utils.grad_compute_context(cfg, gi)
        )
        utils.render_template_to_file(
            'cpg_osqp_grad_compute.h.jinja2', self._include_dir,
            utils.grad_compute_h_context()
        )
        utils.render_template_to_file(
            'cpg_osqp_grad_workspace.h.jinja2', self._include_dir,
            utils.grad_workspace_h_context(cfg)
        )
        utils.write_file(os.path.join(cfg.code_dir, 'c', 'src', 'cpg_osqp_grad_workspace.c'), 'w',
                         self._write_gradient_workspace_def,
                         cfg.prefix, self.canon_gradient.parameter_canon)

        if cfg.gradient_two_stage:
            cg = self.canon_gradient
            cs = self.canon_solver

            utils.write_file(
                os.path.join(self._include_dir, 'cpg_gradient_workspace.h'), 'w',
                utils.write_workspace_prot,
                cfg, cg.prim_variable_info, cg.dual_variable_info,
                cg.parameter_info, cg.parameter_canon, gi, False,
            )
            utils.write_file(
                os.path.join(self._src_dir, 'cpg_gradient_workspace.c'), 'w',
                utils.write_workspace_def,
                cfg, cg.prim_variable_info, cg.dual_variable_info,
                cg.parameter_info, cg.parameter_canon, gi, False,
            )
            utils.write_file(
                os.path.join(self._include_dir, 'cpg_gradient.h'), 'w',
                self._write_gradient_prot,
                cfg,
                cg.prim_variable_info, cg.dual_variable_info,
                cs.prim_variable_info, cs.dual_variable_info,
                cg.parameter_info, cg.parameter_canon, si, gi,
            )
            utils.write_file(
                os.path.join(self._src_dir, 'cpg_gradient.c'), 'w',
                self._write_gradient_def,
                cfg,
                cg.prim_variable_info, cg.dual_variable_info,
                cs.prim_variable_info, cs.dual_variable_info,
                cg.parameter_info, cg.parameter_canon, si, gi
            )
        else:
            utils.write_file(
                os.path.join(self._include_dir, 'cpg_gradient.h'), 'w',
                self._write_gradient_prot,
                cfg, self._pvi, self._dvi, None, None,
                self._pi, self._pc, si, gi
            )
            utils.write_file(
                os.path.join(self._src_dir, 'cpg_gradient.c'), 'w',
                self._write_gradient_def,
                cfg, self._pvi, self._dvi, None, None,
                self._pi, self._pc, si, gi
            )

    def _write_example(self) -> None:
        utils.render_template_to_file(
            'cpg_example.c.jinja2', self._src_dir,
            utils.example_c_context(self.configuration, self._pvi, self._dvi, self._pi)
        )

    def _write_python_module(self) -> None:
        cfg = self.configuration
        si = self.solver_interface
        gi = self.gradient_interface
        utils.render_template_to_file(
            'cpg_module.hpp.jinja2', os.path.join(self._cpp_dir, 'include'),
            utils.module_hpp_context(cfg, self._pi, self._pvi, self._dvi, si, gi)
        )
        utils.write_file(
            os.path.join(self._cpp_dir, 'src', 'cpg_module.cpp'), 'w',
            utils.write_module_def,
            cfg, self._pvi, self._dvi, self._pi, si, gi,
        )

    def _write_python_solver(self) -> None:
        utils.render_template_to_file(
            'cpg_solver.py.jinja2', self.configuration.code_dir,
            utils.solver_py_context(self.configuration, self._pvi, self._dvi, self._pi,
                                    self.solver_interface, self.gradient_interface)
        )

    def _update_cmake(self) -> None:
        cfg = self.configuration

        if not cfg.explicit:
            utils.write_file(
                os.path.join(self._solver_code_dir, 'CMakeLists.txt'), 'a',
                utils.write_canon_cmake,
                'solver', self.solver_interface,
            )

        if cfg.gradient_two_stage:
            if self.solver_interface.has_qdldl:
                # Solver already provides qdldl.c; use set() to list explicit
                # files and avoid the duplicate symbol from file(GLOB *.c)
                osqp_src_replacement = (
                    'set(OSQP_SOURCES\n'
                    '  ${CMAKE_CURRENT_SOURCE_DIR}/src/qdldl_interface.c\n'
                    '  ${CMAKE_CURRENT_SOURCE_DIR}/src/kkt.c)'
                )
                old_block = (
                    'file(\n  GLOB\n  OSQP_SOURCES\n'
                    '  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)'
                )
            else:
                osqp_src_replacement = (
                    '${CMAKE_CURRENT_SOURCE_DIR}/src/qdldl*.c\n'
                    '  ${CMAKE_CURRENT_SOURCE_DIR}/src/kkt.c'
                )
                old_block = '${CMAKE_CURRENT_SOURCE_DIR}/src/*.c'
            utils.read_write_file(
                os.path.join(self._osqp_code_dir, 'CMakeLists.txt'),
                lambda x: x.replace(old_block, osqp_src_replacement),
            )
            utils.write_file(
                os.path.join(self._osqp_code_dir, 'CMakeLists.txt'), 'a',
                utils.write_canon_cmake,
                'osqp', self.gradient_interface,
            )
            utils.read_write_file(
                os.path.join(self._osqp_code_dir, 'src', 'kkt.c'),
                lambda x: x.replace('#include "kkt.h"', '#include "../inc/private/kkt.h"'),
            )
            replacements = [
                ('#include "kkt.h"', '#include "../inc/private/kkt.h"'),
                ('#include "util.h"', '#include "../inc/private/util.h"')
            ]
            utils.read_write_file(
                os.path.join(self._osqp_code_dir, 'src', 'qdldl_interface.c'),
                lambda x: utils.multiple_replace(x, replacements)
            )

    def _update_readme(self) -> None:
        utils.render_template_to_file(
            'README.html.jinja2', self.configuration.code_dir,
            utils.readme_context(self.configuration, self._pvi, self._dvi, self._pi,
                                 self.solver_interface)
        )
