"""
Copyright 2023-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np

from cvxpygen.mappings import PrimalVariableInfo, DualVariableInfo, ParameterCanon


UNSUPPORTED_ON_WINDOWS = ['CLARABEL']


class QPCanonMixin:
    """
    Mixin for solvers that use the OSQP QP canonical form (P, q, d, A, l, u).

    Provides the shared constructor and augment_vector_parameter implementations
    for both OSQPInterface and PDAQPInterface. Both take their inputs from
    CVXPY's canonical maps for OSQP.
    """

    solver_type = 'quadratic'
    canon_p_ids = ['P', 'q', 'd', 'A', 'l', 'u']
    dual_var_split = False
    dual_var_names = ['y']

    def __init__(self, canon_maps, enable_settings):
        n_var = canon_maps.maps['q'][1].size
        n_eq = canon_maps.cone_dims.zero
        n_ineq = canon_maps.cone_dims.nonneg

        super().__init__(self.solver_name, n_var, n_eq, n_ineq, {}, enable_settings)

    def canonical_mapping_rows(self, p_id):
        # l's inequality rows are the constant -inf; augment_vector_parameter restores them.
        return slice(0, self.n_eq) if p_id == 'l' else slice(None)

    def augment_vector_parameter(self, p_id, vector_parameter):
        if p_id == 'l':
            return np.concatenate((vector_parameter, -np.inf * np.ones(self.n_ineq)), axis=0)
        return vector_parameter


class SolverInterface(ABC):

    cvxpy_solver_name: str = None  # CVXPY solver whose canonical maps the interface uses

    def __init__(self, solver_name, n_var, n_eq, n_ineq, canon_constants, enable_settings):
        
        self.solver_name = solver_name
        self.n_var = n_var
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        
        self.canon_constants = canon_constants
        self.enable_settings = enable_settings

        self.configure_settings()

    @property
    @abstractmethod
    def canon_p_ids(self):
        pass

    @property
    @abstractmethod
    def stgs(self):
        pass

    @staticmethod
    def ret_prim_func_exists(variable_info: PrimalVariableInfo) -> bool:
        return any(variable_info.sym) or any([s == 1 for s in variable_info.name_to_size.values()])

    @staticmethod
    def ret_dual_func_exists(dual_variable_info: DualVariableInfo) -> bool:
        return any([s == 1 for s in dual_variable_info.name_to_size.values()])

    def configure_settings(self) -> None:
        for s in self.stgs:
            if s in self.enable_settings:
                self.stgs[s].enabled = True
        for s in set(self.enable_settings)-set(self.stgs):
            warnings.warn(f'Cannot enable setting {s} for solver {self.solver_name}')

    def augment_vector_parameter(self, p_id, vector_parameter):
        return vector_parameter

    def canonical_mapping_rows(self, p_id):
        """Rows of the canonical map for p_id that are canonical parameter entries."""
        return slice(None)
    
    @property
    def stgs_names_enabled(self):
        return [n for n, s in self.stgs.items() if s.enabled]

    @property
    def stgs_names_to_type(self):
        return {n: s.type for n, s in self.stgs.items() if s.enabled}
    
    @property
    def stgs_names_to_default(self):
        return {n: s.default for n, s in self.stgs.items() if s.enabled}
    
    @property
    def stgs_translation(self):
        return {s.name_cvxpy: n for n, s in self.stgs.items() if s.enabled and s.name_cvxpy is not None}
    
    @staticmethod
    def check_unsupported_cones(cone_dims: "ConeDims") -> None:
        pass

    def cmake_context_extra(self) -> dict:
        """Solver-specific additions to the top-level c/CMakeLists.txt context.

        Solvers override this to express extra include directories, extra target_link_libraries calls, etc.
        The solver's own CMakeLists.txt is edited directly in generate_code().
        """
        return {
            'solver_code_cmake_include_dir': '${CMAKE_CURRENT_SOURCE_DIR}/solver_code/include',
            'extra_cmake_include_dirs': [],
            'packages': [],
            'cmake_target_link_libs': [],
            'cmake_definitions': [],
        }

    @abstractmethod
    def generate_code(self, configuration, code_dir, solver_code_dir, cvxpygen_directory,
                      parameter_canon: ParameterCanon, gradient, prefix) -> None:
        pass

    def declare_workspace(self, f, prefix, parameter_canon) -> None:
        pass

    def define_workspace(self, f, prefix, parameter_canon) -> None:
        pass
    
    def write_gradient_def(f, configuration,
                           variable_info_first, dual_variable_info_first,
                           variable_info_second, dual_variable_info_second,
                           parameter_info, parameter_canon, solver_interface) -> None:
        pass
    
    def write_gradient_prot(f, configuration,
                            variable_info_first, dual_variable_info_first,
                            variable_info_second, dual_variable_info_second,
                            parameter_info, parameter_canon, solver_interface) -> None:
        pass
    
    def write_gradient_workspace_def(f, prefix, parameter_canon) -> None:
        pass
    
    def special_settings(self, config) -> dict:
        return {}
