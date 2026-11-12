"""
Copyright 2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import numpy as np
import scipy.sparse as sp
from cvxpygen.mappings import Canon
from cvxpygen.solvers._interface import SolverInterface
from cvxpygen.utils import symmetric_from_upper


class PreSolver:
    """
    Very basic presolver before explicit solver generation.
    """

    def __init__(self) -> None:
        pass
    
    def solve(
        self,
        canon: Canon,
        solver_interface: SolverInterface
    ) -> None:
        
        # check that P and A are constants
        pc = canon.parameter_canon
        if pc.p_id_to_changes['P'] or pc.p_id_to_changes['A']:
            raise ValueError(f'Explicit solver generation: Matrices must be constant!')
        
        # identify to-eliminate vars
        P = symmetric_from_upper(pc.p['P']).toarray()
        q = pc.p['q']
        A = pc.p['A'].toarray()
        u = pc.p['u']
        
        # get indices of vars that do not appear in objective
        not_in_obj = set(np.where(np.all(P == 0, axis=0) & (q == 0))[0])
        
        # get indices of user-defined vars (will not be eliminated)
        pvi = canon.prim_variable_info
        user_defined = set()
        for name, offset in pvi.name_to_offset.items():
            user_defined.update(range(offset, offset + pvi.name_to_size[name]))

        # record to-eliminate vars (not in obj or user-defined, upper-bounded only by non-vars)
        candidate_indices = not_in_obj - user_defined
        to_eliminate = {}  # var_index: (coefficient, constraint_index)
        for i in candidate_indices:
            mask = (A[:, i] > 0)
            if sum(mask) == 1:  # unclear how to handle multiple upper bounds in the parametrized setting
                A_pos_coef = A[mask]
                if np.all(A_pos_coef[:, :i] == 0) & np.all(A_pos_coef[:, i+1:] == 0):
                    to_eliminate[i] = (A_pos_coef[0, i], np.where(mask)[0][0])
        
        # remove variables and respective constraints
        m, n = A.shape
        row_mask = np.ones(m, dtype=bool)
        col_mask = np.ones(n, dtype=bool)
        for i, (coef, constr_ind) in to_eliminate.items():
            row_mask[constr_ind] = False
            col_mask[i] = False
            for k in range(m):
                if k == constr_ind:
                    continue
                if A[k, i] != 0:
                    ratio = A[k, i] / coef
                    pc.p_id_to_mapping['u'][k] -= ratio * pc.p_id_to_mapping['u'][constr_ind]
                    u[k] -= ratio * u[constr_ind]
        
        # update canonical parameters and mappings
        pc.p['P'] = sp.csc_matrix(np.triu(P[np.ix_(col_mask, col_mask)]))
        pc.p['q'] = q[col_mask]
        pc.p['A'] = sp.csc_matrix(A[np.ix_(row_mask, col_mask)])
        pc.p['u'] = u[row_mask]

        pc.p_id_to_mapping['q'] = pc.p_id_to_mapping['q'][col_mask]
        pc.p_id_to_mapping['u'] = pc.p_id_to_mapping['u'][row_mask]

        pc.p_id_to_size['q'] = len(pc.p['q'])
        pc.p_id_to_size['u'] = len(pc.p['u'])

        # index book-keeping for primal variable retrieval
        new_cols = np.cumsum(col_mask) - 1
        for name in list(pvi.name_to_offset):
            pvi.name_to_offset[name] = new_cols[pvi.name_to_offset[name]]
            pvi.name_to_indices[name] = new_cols[pvi.name_to_indices[name]]

        # index book-keeping for dual variable retrieval
        dvi = canon.dual_variable_info
        new_rows = np.cumsum(row_mask) - 1
        for name in list(dvi.name_to_offset):
            dvi.name_to_offset[name] = new_rows[dvi.name_to_offset[name]]
            vec, old_indices = dvi.name_to_indices[name]
            dvi.name_to_indices[name] = (vec, new_rows[old_indices])

        # update solver interface   
        solver_interface.n_var = pc.p_id_to_size['q']
        solver_interface.n_ineq -= len(to_eliminate)
