"""
Copyright 2021-2026 Maximilian Schaller and the CVXPYgen contributors
Licensed under the Apache License, Version 2.0
"""

import os
import numpy as np
import scipy.sparse as sp
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_env = None


def _get_env() -> Environment:
    global _env
    if _env is None:
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        _env = Environment(
            loader=FileSystemLoader(templates_dir),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _env


def render_template(template_name: str, context: dict) -> str:
    """Render a Jinja2 template from the templates/ directory and return the result as a string."""
    return _get_env().get_template(template_name).render(**context)


def render_template_to_file(template_name: str, output_dir: str, context: dict = {}, ) -> None:
    """Render a Jinja2 template and write the result to output_path."""
    content = render_template(template_name, context)
    output_filename = template_name.split('-')[-1].replace('.jinja2', '')
    with open(os.path.join(output_dir, output_filename), 'w', encoding='utf-8') as f:
        f.write(content)


def write_file(path, mode, function, *args):
    """Write data to a file using a specific utility function."""
    with open(path, mode) as file:
        function(file, *args)
    

def read_write_file(path, function, *args):
    """Read data from a file, process it, and write back."""
    with open(path, 'r') as file:
        data = file.read()
    data = function(data, *args)
    with open(path, 'w') as file:
        file.write(data)


def multiple_replace(text, replacements):
    """Perform multiple replacements (list of 2-tuples) on text"""
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def cut_from_expr(text, expr):
    """Cut file contents from a specific expression"""
    idx = text.find(expr)
    
    if idx != -1:
        return text[:idx]
    else:
        return text
    
    
def type_to_cast(typ, qualifier=''):
    """Convert type to C type cast"""
    
    qualifier = '' if qualifier == '' else f'{qualifier}_'
    
    if typ == 'float':
        return f'(cpg_{qualifier}float *) '
    elif typ == 'int':
        return f'(cpg_{qualifier}int *) '
    else:
        return ''


def write_vec_def(f, vec, name, typ, qualifier=''):
    """
    Write vector to file
    """
    
    qualifier = '' if qualifier == '' else f'{qualifier}_'
    if typ == 'cpg_float':
        typ = f'cpg_{qualifier}float'
    elif typ == 'cpg_int':
        typ = f'cpg_{qualifier}int'
    
    f.write(f'{typ} {name}[{len(vec)}] = {{\n')

    # Write vector components
    for i in range(len(vec)):
        if 'float' in typ:
            f.write(f'({typ})%.20f,\n' % vec[i])
        else:
            f.write(f'{vec[i]},\n')

    f.write('};\n')


def write_vec_prot(f, vec, name, typ):
    """
    Write vector to file
    """
    f.write(f'extern {typ} {name}[{len(vec)}];\n')


def write_mat_def(f, mat, name, qualifier=''):
    """
    Write sparse matrix (scipy compressed sparse column) to file
    """
    
    qualifier = '' if qualifier == '' else f'{qualifier}_'
    
    write_vec_def(f, mat.indices, name + '_i', f'cpg_{qualifier}int')
    write_vec_def(f, mat.indptr, name + '_p', f'cpg_{qualifier}int')
    write_vec_def(f, mat.data, name + '_x', f'cpg_{qualifier}float')

    f.write(f'cpg_{qualifier}csc {name} = {{')
    f.write(f'{name}_p, ')
    f.write(f'{name}_i, ')
    f.write(f'{name}_x, {mat.nnz}}};\n')


def write_L_def(f, N, name, qualifier=''):
    """
    Write sparse lower triangular matrix (scipy compressed sparse column) to file
    and allocate enough memory for maximum fill-in of the lower triangle
    where the diagonal is implicitly ones
    """
    
    # create scipy csc matrix with all ones below the diagonal, without the diagonal
    indices = []
    data = []
    indptr = [0]
    for i in range(N):
        indices.extend(range(i + 1, N))
        data.extend(np.ones(N - i - 1))
        indptr.append(len(indices))
    mat = sp.csc_matrix((data, indices, indptr), shape=(N, N))

    write_mat_def(f, mat, name, qualifier)


def write_mat_prot(f, mat, name):
    """
    Write sparse matrix (scipy compressed sparse column) to file
    """
    f.write(f'extern cpg_csc {name};\n')


def write_dense_mat_def(f, mat, name):
    """
    Write dense matrix to file
    """

    f.write(f'cpg_float {name}[{mat.size}] = {{\n')

    # represent matrix as vector (Fortran style)
    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            f.write('(cpg_float)%.20f,\n' % mat[i, j])

    f.write('};\n')


def write_dense_mat_prot(f, mat, name):
    """
    Write dense matrix to file
    """

    f.write(f'extern cpg_float cpg_{name}[{mat.size}];\n')


def symmetric_from_upper(P_upper):
    """Full symmetric matrix from its upper triangle (the canonical form of P)."""
    P_upper = sp.csc_matrix(P_upper)
    return sp.csc_matrix(P_upper + sp.triu(P_upper, k=1).T)


def write_description(f, file_type, content):
    """
    Timestamp and file content to beginning of file
    """

    if file_type == 'py':
        comment_str = '"""'
    else:
        comment_str = '/*'
    now = datetime.now()
    f.write(f'\n{comment_str}\n')
    f.write(f'Auto-generated by CVXPYgen {now.strftime("on %B %d, %Y at %H:%M:%S")}\n')
    f.write(f'Content: {content}\n')
    f.write('Copyright 2021-2026 Maximilian Schaller and the CVXPYgen contributors\n')
    f.write('Licensed under the Apache License, Version 2.0\n')
    f.write(f'{comment_str[::-1]}\n\n')


def replace_in_file(filepath, replacements):
    """
    Replace strings in file
    """

    with open(filepath, 'r') as f:
        t = f.read()
    for old, new in replacements:
        t = t.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(t)


def replace_inf(v):
    """
    Replace infinity by large number
    """

    # check if dealing with csc dict or numpy array
    if type(v) == sp.csc_matrix:
        sign = np.sign(v.data)
        idx = np.isinf(v.data)
        v.data[idx] = 1e30 * sign[idx]
    else:
        sign = np.sign(v)
        idx = np.isinf(v)
        v[idx] = 1e30 * sign[idx]

    return v


def param_is_empty(param):
    """
    Check if parameter is empty
    """

    if type(param) == dict:
        return param['x'].size == 0
    else:
        return param.size == 0


def is_mathematical_scalar(x):
    """
    Check if input is a scalar in mathematical sense, e.g., returning True for vectors with length 1
    """

    return True if np.isscalar(x) else x.size == 1


def squeeze_scalar(val):
    """
    Turn size-1 numpy arrays into scalars
    """
    if isinstance(val, np.ndarray):
        val = val.squeeze()
        if val.shape == ():
            return val.item()
    return val


def ldl(A):
    
    n = A.shape[0]
    L = np.zeros_like(A)
    D = np.zeros(n)
    
    for i in range(n):
        for j in range(i):
            sum_LDL = np.dot(L[i, :j], L[j, :j] * D[:j])
            L[i, j] = (A[i, j] - sum_LDL) / D[j]
        
        sum_LDL = np.dot(L[i, :i], L[i, :i] * D[:i])
        D[i] = A[i, i] - sum_LDL
        L[i, i] = 1
    
    return L, D


def write_canonicalize(f, canon_name, s, mapping, prefix, prefix_params=None):
    """
    Write function to compute canonical parameter value
    """
    
    if prefix_params is None:
        prefix_params = prefix
        
    indexing = '' if canon_name == 'd' else '[i]'

    f.write(f'  for(i=0; i<{mapping.shape[0]}; i++){{\n')
    f.write(f'    {prefix}Canon_Params.{canon_name}{s}{indexing} = 0;\n')
    f.write(f'    for(j={prefix}canon_{canon_name}_map.p[i]; j<{prefix}canon_{canon_name}_map.p[i+1]; j++){{\n')
    f.write(f'      {prefix}Canon_Params.{canon_name}{s}{indexing} += {prefix}canon_{canon_name}_map.x[j]*{prefix_params}cpg_params_vec[{prefix}canon_{canon_name}_map.i[j]];\n')
    f.write(f'    }}\n')
    f.write(f'  }}\n')


def write_param_def(f, param, name, prefix, suffix):
    """
    Write vectors and matrices
    """
    if not param_is_empty(param):
        if name.isupper():
            write_mat_def(f, param, f'{prefix}canon_{name}{suffix}')
        elif name == 'd':
            f.write(f'cpg_float {prefix}canon_d{suffix} = %.20f;\n' % replace_inf(param[0]))
        else:
            write_vec_def(f, param, f'{prefix}canon_{name}{suffix}', 'cpg_float')
        f.write('\n')


def write_param_prot(f, param, name, prefix, suffix):
    """
    Write vectors and matrices
    """
    if not param_is_empty(param):
        if name.isupper():
            write_mat_prot(f, param, f'{prefix}canon_{name}{suffix}')
        elif name == 'd':
            f.write(f'extern cpg_float {prefix}canon_d{suffix};\n')
        else:
            write_vec_prot(f, param, f'{prefix}canon_{name}{suffix}', 'cpg_float')


def write_struct_def(f, fields, casts, values, name, typ):
    """
    Write structure to file
    """

    f.write(f'{typ} {name} = {{\n')

    # write structure fields
    for field, cast, value in zip(fields, casts, values):
        if value in ['0', 'SCS_NULL']:
            cast = ''
        f.write(f'.{field} = {cast}{value},\n')

    f.write('};\n')


def write_struct_prot(f, name, typ):
    """
    Write structure to file
    """

    f.write(f'extern {typ} {name};\n')


def extend_functions_if_false(pus, functions_if_false):
    """
    Recursively extends functions_if_false in the parameter update structure
    """

    if functions_if_false is None:
        return []

    extended_functions_if_false = []

    for function in functions_if_false:
        extended_functions_if_false.append(function)
        extended_functions_if_false.extend(extend_functions_if_false(pus, pus[function].update_pending_logic.functions_if_false))

    return extended_functions_if_false


def remove_function(functions, function_to_remove):
    if function_to_remove in functions:
        functions.remove(function_to_remove)
    return functions


def analyze_pus(pus, p_id_to_changes):
    '''
    Analyze parameter update structure (pus) to return set of canonical update functions
    that are to be written to code without logical inter-connections and functions that
    are never called
    '''
    
    functions = set(pus.keys())
    functions_called = set(pus.keys())
    functions_secondary = []
    
    for function, logic in pus.items():
        
        up_logic = logic.update_pending_logic
        operator = up_logic.operator
        functions_if_false = extend_functions_if_false(pus, up_logic.functions_if_false)
            
        if operator in ['&&', '&', 'and', 'AND']:
            skip = False
            for p in up_logic.parameters_outdated:
                if not p_id_to_changes.get(p, False):
                    functions_called = remove_function(functions_called, function)
                    skip = True
            if skip:
                continue
        elif operator in ['||', '|', 'or', 'OR']:
            skip = True
            for p in up_logic.parameters_outdated:
                if p_id_to_changes.get(p, False):
                    skip = False
            if skip:
                functions_called = remove_function(functions_called, function)
                continue
        elif operator is None:
            if up_logic.extra_condition_operator is None and len(up_logic.parameters_outdated) == 1 and not p_id_to_changes[function]:
                functions_called = remove_function(functions_called, function)
                continue
        else:
            raise ValueError(f'Operator "{operator}" not implemented.')
        
        if functions_if_false is None:
            continue
            
        for f in functions_if_false:
            if f not in functions:
                raise ValueError(f'"{f}" is not part of parameter update structure.')
        functions_secondary.extend(functions_if_false)
                    
    return functions_called-set(functions_secondary), functions-functions_called


def write_update_structure(f, config, parameter_canon, pus, functions, functions_never_called, depth=0):
    """
    Recursively write logical parameter update structure to file
    """
    
    operator_map = {'&&': '&&', '&': '&&', 'and': '&&', 'AND': '&&',
                    '||': '||', '|': '||', 'or': '||', 'OR': '||'}

    if functions is None:
        return

    write_else = depth > 0 and len(set(functions) - set(functions_never_called)) > 0

    if write_else:
        f.write(' else {\n')

    if 'init' in functions:
        functions.remove('init')
        functions = ['init'] + list(functions)

    for function in functions:

        logic = pus[function]
        up_logic = logic.update_pending_logic
        if 'P' in up_logic.parameters_outdated and not parameter_canon.quad_obj:
            up_logic.parameters_outdated.remove('P')

        if function not in functions_never_called:
            extra_condition = f'{up_logic.extra_condition.format(prefix=config.prefix)} ' if up_logic.extra_condition is not None else ''
            extra_condition_operator = f' {operator_map.get(up_logic.extra_condition_operator, "||")} ' if up_logic.extra_condition is not None and len(up_logic.parameters_outdated) > 0 else ''
            parameters_outdated = f'{(" "+operator_map.get(up_logic.operator, "&&")+" ").join([f"{config.prefix}Canon_Outdated.{p}" for p in up_logic.parameters_outdated])}'

            f.write(f'{"  "*(depth+1)}if ({extra_condition}{extra_condition_operator}{parameters_outdated}) {{\n')
            f.write(f'{"  "*(depth+2)}{logic.function_call.format(prefix=config.prefix)};\n')
            f.write(f'{"  "*(depth+1)}}}')
            new_depth = depth + 1
        else:
            new_depth = depth * 1

        write_update_structure(f, config, parameter_canon, pus, up_logic.functions_if_false, functions_never_called, new_depth)

        if function not in functions_never_called:
            f.write('\n')

    if write_else:
        f.write(f'{"  " * depth}}}')


def write_workspace_def(f, config, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface, full=True):

    if full:
        prefix = config.prefix
    else:
        prefix = f'gradient_{config.prefix}'

    write_description(f, 'c', 'Variable definitions')
    f.write('#include "cpg_workspace.h"\n')
    if config.gradient_two_stage:
        f.write('#include "cpg_gradient.h"\n')
        f.write('#include "cpg_gradient_workspace.h"\n')

    if full:
        f.write('\n// Vector containing flattened user-defined parameters\n')
        write_vec_def(f, parameter_info.flat_usp, f'{prefix}cpg_params_vec', 'cpg_float')
        if config.explicit:
            write_vec_def(f, parameter_info.lower, f'{prefix}cpg_params_vec_lower', 'cpg_float')
            write_vec_def(f, parameter_info.upper, f'{prefix}cpg_params_vec_upper', 'cpg_float')
    
    f.write('\n// Sparse mappings from user-defined to canonical parameters\n')
    for p_id, mapping in parameter_canon.p_id_to_mapping.items():
        if parameter_canon.p_id_to_changes[p_id]:
            write_mat_def(f, mapping, f'{prefix}canon_{p_id}_map')
            f.write('\n')

    p_ids = list(parameter_canon.p.keys())

    if config.explicit:
        p_ids.remove('A') # Constraints not needed for explicit
        if variable_info.reduced:
            p_ids.remove('P') # Objective cannot be computed if variables have  been removed

    canon_casts = []
    f.write('// Canonical parameters\n')
    for p_id in p_ids:
        p = parameter_canon.p[p_id]
        if p_id == 'd':
            canon_casts.append('')
        else:
            write_param_def(f, replace_inf(p), p_id, prefix, '')
            if solver_interface.inmemory_preconditioning:
                write_param_def(f, replace_inf(p), p_id, prefix, '_conditioning')
            if p_id.isupper():
                canon_casts.append('')
            else:
                canon_casts.append('(cpg_float *) ')
                
    if config.explicit:
        write_vec_def(f, np.zeros(parameter_canon.n_param_reduced), f'{config.prefix}cpg_theta', 'cpg_float')
        if variable_info.reduced:
            write_vec_def(f, np.zeros(sum(variable_info.name_to_size_reduced.values())), 'sol_x', 'cpg_float')
        else:
            write_vec_def(f, np.zeros(solver_interface.n_var), 'sol_x', 'cpg_float')
    if config.explicit == 2:
        write_vec_def(f, np.zeros(parameter_canon.n_dual_reduced), 'sol_y', 'cpg_float')

    f.write('// Struct containing canonical parameters\n')

    struct_values = []
    struct_values_conditioning = []
    for i, p_id in enumerate(p_ids):
        p = parameter_canon.p[p_id]
        if type(p) == sp.csc_matrix:
            length = len(p.data)
        else:
            length = len(p)
        if length == 0:
            struct_values.append('0')
            if solver_interface.inmemory_preconditioning:
                struct_values_conditioning.append('0')
        elif p_id=='d':
            struct_values.append('%.20f' % squeeze_scalar(p))
            if solver_interface.inmemory_preconditioning:
                struct_values_conditioning.append('%.20f' % squeeze_scalar(p))
        else:
            struct_values.append(f'&{prefix}canon_{p_id}')
            if solver_interface.inmemory_preconditioning:
                struct_values_conditioning.append(f'&{prefix}canon_{p_id}_conditioning')

    write_struct_def(f, p_ids, canon_casts, struct_values, f'{prefix}Canon_Params', f'Canon_Params_{"" if full else "Gradient_"}t')
    f.write('\n')
    if solver_interface.inmemory_preconditioning:
        write_struct_def(f, p_ids, canon_casts, struct_values_conditioning, f'{prefix}Canon_Params_conditioning', f'Canon_Params_{"" if full else "Gradient_"}t')
        f.write('\n')

    # Boolean struct for outdated parameter flags
    if full:
        f.write('// Struct containing flags for outdated canonical parameters\n')
        f.write(f'Canon_Outdated_{"" if full else "Gradient_"}t {prefix}Canon_Outdated = {{\n')
        for p_id in parameter_canon.p.keys():
            f.write(f'.{p_id} = 1,\n')
        f.write('};\n\n')
    if (config.gradient and not config.gradient_two_stage) or (not full and config.gradient_two_stage):
        f.write('// Struct containing flags for outdated canonical parameters for gradient computation\n')
        f.write(f'Canon_Outdated_{"Gradient_" if not full else ""}t {prefix}Canon_Outdated_Grad = {{\n')
        for p_id in parameter_canon.p.keys():
            f.write(f'.{p_id} = {"0" if full else "1"},\n')
        f.write('};\n\n')

    if not full:
        return
    
    prim_cast = []
    if any(variable_info.name_to_sym) or not solver_interface.sol_statically_allocated:
        f.write('// User-defined variables\n')
    for name, value in variable_info.name_to_init.items():
        if is_mathematical_scalar(value):
            prim_cast.append('')
        else:
            prim_cast.append('(cpg_float *) ')
            if variable_info.name_to_sym[name] or not solver_interface.sol_statically_allocated:
                write_vec_def(f, value.flatten(order='F'), prefix + name, 'cpg_float')
                f.write('\n')

    # Dual solution pointer is always overridden to gradient_{prefix}sol_y when
    # gradient_two_stage=True, so it never needs the solver-level prefix.
    # Primal solution pointer is only overridden (to gradient_{prefix}sol_x) when
    # gradient_two_stage=True AND ret_prim_func_exists=True; when ret_prim_func_exists
    # is False (e.g., SCS), the original solver variable (e.g., scs_x) is kept and
    # DOES need the solver-level prefix.
    dual_result_prefix = prefix if not solver_interface.ws_statically_allocated_in_solver_code and not config.gradient_two_stage else ''
    prim_overridden = config.gradient_two_stage and solver_interface.ret_prim_func_exists(variable_info)
    prim_result_prefix = prefix if not solver_interface.ws_statically_allocated_in_solver_code and not prim_overridden else ''

    f.write('// Struct containing primal solution\n')
    CPG_Prim_fields = list(variable_info.name_to_init.keys())
    CPG_Prim_values = []
    for name, var in variable_info.name_to_init.items():
        offset = variable_info.name_to_offset[name]
        if variable_info.name_to_size[name] == 1:
            CPG_Prim_values.append('0')
        else:
            if variable_info.name_to_sym[name] or not solver_interface.sol_statically_allocated:
                CPG_Prim_values.append('&' + prefix + name)
            else:
                CPG_Prim_values.append(f'&{prim_result_prefix}{solver_interface.ws_ptrs.primal_solution} + {offset}')
    write_struct_def(f, CPG_Prim_fields, prim_cast, CPG_Prim_values, f'{prefix}CPG_Prim', 'CPG_Prim_t')

    if config.explicit != 1:
        if len(dual_variable_info.name_to_init) > 0:
            dual_cast = []
            if not solver_interface.sol_statically_allocated:
                f.write('\n// Dual variables associated with user-defined constraints\n')
            for name, value in dual_variable_info.name_to_init.items():
                if is_mathematical_scalar(value):
                    dual_cast.append('')
                else:
                    dual_cast.append('(cpg_float *) ')
                    if not solver_interface.sol_statically_allocated:
                        write_vec_def(f, value, prefix + name, 'cpg_float')
                        f.write('\n')

            f.write('// Struct containing dual solution\n')
            CPG_Dual_fields = dual_variable_info.name_to_init.keys()
            CPG_Dual_values = []
            for name, var in dual_variable_info.name_to_init.items():
                vec = dual_variable_info.name_to_vec[name]
                offset = dual_variable_info.name_to_offset[name]
                if is_mathematical_scalar(var):
                    CPG_Dual_values.append('0')
                else:
                    if not solver_interface.sol_statically_allocated:
                        CPG_Dual_values.append('&' + prefix + name)
                    else:
                        CPG_Dual_values.append(f'&{dual_result_prefix}{solver_interface.ws_ptrs.dual_solution.format(dual_var_name=vec)} + {offset}')
            write_struct_def(f, CPG_Dual_fields, dual_cast, CPG_Dual_values, f'{prefix}CPG_Dual', 'CPG_Dual_t')

    if not config.explicit:
        f.write('\n// Struct containing solver info\n')
        CPG_Info_fields = ['obj_val', 'iter', 'status', 'pri_res', 'dua_res']
        CPG_Info_values = ['0', '0', ('0' if solver_interface.status_is_int else '"unknown"'), '0', '0']
        info_cast = ['', '', '', '', '']
        write_struct_def(f, CPG_Info_fields, info_cast, CPG_Info_values, f'{prefix}CPG_Info', 'CPG_Info_t')

    f.write('\n// Struct containing solution and info\n')
    if config.explicit == 1:
        CPG_Result_fields = ['prim']
        result_cast = ['']
        CPG_Result_values = [f'&{prefix}CPG_Prim']
    elif config.explicit == 2:
        if len(dual_variable_info.name_to_init) > 0:
            CPG_Result_fields = ['prim', 'dual']
            result_cast = ['', '']
            CPG_Result_values = [f'&{prefix}CPG_Prim', f'&{prefix}CPG_Dual']
        else:
            CPG_Result_fields = ['prim']
            result_cast = ['']
            CPG_Result_values = [f'&{prefix}CPG_Prim']
    else:
        if len(dual_variable_info.name_to_init) > 0:
            CPG_Result_fields = ['prim', 'dual', 'info']
            result_cast = ['', '', '']
            CPG_Result_values = [f'&{prefix}CPG_Prim', f'&{prefix}CPG_Dual',
                                f'&{prefix}CPG_Info']
        else:
            CPG_Result_fields = ['prim', 'info']
            result_cast = ['', '']
            CPG_Result_values = [f'&{prefix}CPG_Prim', f'&{prefix}CPG_Info']
    write_struct_def(f, CPG_Result_fields, result_cast, CPG_Result_values, f'{prefix}CPG_Result', 'CPG_Result_t')

    if not config.explicit:  # TODO: explicit case
        if solver_interface.stgs_requires_extra_struct_type:
            f.write('\n// Struct containing solver settings\n')
            f.write(f'Canon_Settings_t {prefix}Canon_Settings = {{\n')
            for name, default in solver_interface.stgs_names_to_default.items():
                f.write(f'.{name} = {default},\n')
            f.write('};\n')

    if not solver_interface.ws_statically_allocated_in_solver_code:
        solver_interface.define_workspace(f, prefix, parameter_canon)
        
    if config.gradient:
        
        f.write('\n// Derivative workspace\n')
        write_vec_def(f, np.zeros(len(parameter_info.flat_usp)), f'{prefix}cpg_dp', 'cpg_float')
        
        delta_cast = []
        f.write('// User-defined parameter deltas\n')
        for col, name in parameter_info.col_to_name_usp.items():
            delta_cast.append('(cpg_float *) ')
        
        f.write('\n// Struct containing parameter deltas\n')
        CPG_Delta_fields = list(parameter_info.col_to_name_usp.values())
        CPG_Delta_values = []
        for col, name in parameter_info.col_to_name_usp.items():
            CPG_Delta_values.append(f'&{prefix}cpg_dp + {col}')
        write_struct_def(f, CPG_Delta_fields, delta_cast, CPG_Delta_values, f'{prefix}CPG_Delta', 'CPG_Delta_t')


def write_workspace_prot(f, config, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface, full=True):
    """"
    Write workspace initialization to file
    """

    prefix = config.prefix if full else f'gradient_{config.prefix}'

    write_description(f, 'c', 'Type definitions and variable declarations')
    if full:
        for header_file in solver_interface.header_files:
            f.write(f'#include {header_file}\n')
    else:
        f.write('#include "cpg_workspace.h"\n\n')

    if full:
        # definition safeguard
        f.write('\n#ifndef CPG_TYPES_H\n')
        f.write('# define CPG_TYPES_H\n\n')

        if config.explicit:
            f.write(f'typedef float cpg_float;\n')
        else:
            f.write(f'typedef {solver_interface.numeric_types["float"]} cpg_float;\n')
        f.write(f'typedef {solver_interface.numeric_types["int"]} cpg_int;\n\n')

        # struct definitions
        f.write('// Compressed sparse column matrix\n')
        f.write('typedef struct {\n')
        f.write('  cpg_int      *p;\n')
        f.write('  cpg_int      *i;\n')
        f.write('  cpg_float    *x;\n')
        f.write('  cpg_int      nnz;\n')
        f.write('} cpg_csc;\n\n')

    f.write('// Canonical parameters\n')
    f.write('typedef struct {\n')
    for p_id in parameter_canon.p.keys():
        if p_id.isupper():
            f.write(f'  cpg_csc      *{(p_id+";").ljust(8)}   // Canonical parameter {p_id}\n')
        else:
            if p_id == 'd':
                s = ''
            else:
                s = '*'
            f.write(f'  cpg_float    {(s+p_id+";").ljust(9)}   // Canonical parameter {p_id}\n')
    f.write(f'}} Canon_Params_{"" if full else "Gradient_"}t;\n\n')

    f.write('// Flags indicating outdated canonical parameters\n')
    f.write('typedef struct {\n')
    for p_id in parameter_canon.p.keys():
        f.write(f'  int        {(p_id + ";").ljust(10)}    // Bool, if canonical parameter {p_id} outdated\n')
    f.write(f'}} Canon_Outdated_{"" if full else "Gradient_"}t;\n\n')

    if full:
        f.write('// Primal solution\n')
        f.write('typedef struct {\n')
        for name, size in variable_info.name_to_size.items():
            if size == 1:
                s = ''
            else:
                s = '*'
            f.write(f'  cpg_float    {(s + name + ";").ljust(9)}   // Your variable {name}\n')
        f.write('} CPG_Prim_t;\n\n')

        if len(dual_variable_info.name_to_init) > 0:
            f.write('// Dual solution\n')
            f.write('typedef struct {\n')
            for name, size in dual_variable_info.name_to_size.items():
                if size == 1:
                    decl = f'{name};'
                else:
                    decl = f'*{name};'
                f.write(f'  cpg_float    {decl.ljust(9)}   // Your dual variable for constraint {name}\n')
            f.write('} CPG_Dual_t;\n\n')

        if not config.explicit:
            f.write('// Solver information\n')
            f.write('typedef struct {\n')
            f.write('  cpg_float    obj_val;    // Objective function value\n')
            f.write('  cpg_int      iter;       // Number of iterations\n')
            f.write(f'  {"cpg_int      status;     " if solver_interface.status_is_int else "char         *status;    "}// Solver status\n')
            f.write('  cpg_float    pri_res;    // Primal residual\n')
            f.write('  cpg_float    dua_res;    // Dual residual\n')
            f.write('} CPG_Info_t;\n\n')

        f.write('// Solution and solver information\n')
        f.write('typedef struct {\n')
        f.write('  CPG_Prim_t *prim;        // Primal solution\n')
        if config.explicit != 1:
            if len(dual_variable_info.name_to_init) > 0:
                f.write('  CPG_Dual_t *dual;        // Dual solution\n')
        if not config.explicit:
            f.write('  CPG_Info_t *info;        // Solver info\n')
        f.write('} CPG_Result_t;\n\n')

        f.write('// Parameter deltas\n')
        f.write('typedef struct {\n')
        for name, size in parameter_info.name_to_size_usp.items():
            f.write(f'  cpg_float    {("*" + name + ";").ljust(9)}   // Delta of your parameter {name}\n')
        f.write('} CPG_Delta_t;\n\n')

        if solver_interface.stgs_requires_extra_struct_type:
            f.write('// Solver settings\n')
            f.write('typedef struct {\n')
            for name, typ in solver_interface.stgs_names_to_type.items():
                f.write(f'  {typ.ljust(11)}{name};\n')
            f.write('} Canon_Settings_t;\n\n')

    if full:
        f.write('#endif // ifndef CPG_TYPES_H\n')
        f.write('\n// Vector containing flattened user-defined parameters\n')
        write_vec_prot(f, parameter_info.flat_usp, f'{prefix}cpg_params_vec', 'cpg_float')
        if config.explicit:
            write_vec_prot(f, parameter_info.lower, f'{prefix}cpg_params_vec_lower', 'cpg_float')
            write_vec_prot(f, parameter_info.upper, f'{prefix}cpg_params_vec_upper', 'cpg_float')

    f.write('\n// Sparse mappings from user-defined to canonical parameters\n')
    for p_id, mapping in parameter_canon.p_id_to_mapping.items():
        if parameter_canon.p_id_to_changes[p_id]:
            write_mat_prot(f, mapping, f'{prefix}canon_{p_id}_map')

    f.write('\n// Canonical parameters\n')
    for p_id, p in parameter_canon.p.items():
        if p_id != 'd':
            write_param_prot(f, p, p_id, f'{prefix}', '')
            if solver_interface.inmemory_preconditioning:
                write_param_prot(f, p, p_id, f'{config.prefix}', '_conditioning')

    if config.explicit:
        write_vec_prot(f, np.zeros(parameter_canon.n_param_reduced), f'{prefix}cpg_theta', 'cpg_float')
        if variable_info.reduced:
            write_vec_prot(f, np.zeros(sum(variable_info.name_to_size_reduced.values())), 'sol_x', 'cpg_float')
        else:
            write_vec_prot(f, np.zeros(solver_interface.n_var), 'sol_x', 'cpg_float')
    if config.explicit == 2:
        write_vec_prot(f, np.zeros(parameter_canon.n_dual_reduced), 'sol_y', 'cpg_float')

    f.write('\n// Struct containing canonical parameters\n')
    write_struct_prot(f, f'{prefix}Canon_Params', f'Canon_Params_{"" if full else "Gradient_"}t')
    if solver_interface.inmemory_preconditioning:
        write_struct_prot(f, f'{prefix}Canon_Params_conditioning', f'Canon_Params_{"" if full else "Gradient_"}t')

    if full:
        f.write('\n// Struct containing flags for outdated canonical parameters\n')
        f.write(f'extern Canon_Outdated_{"" if full else "Gradient_"}t {prefix}Canon_Outdated;\n')
    if (config.gradient and not config.gradient_two_stage) or (not full and config.gradient_two_stage):
        f.write(f'extern Canon_Outdated_{"" if full else "Gradient_"}t {prefix}Canon_Outdated_Grad;\n')

    if full:
        if any(variable_info.name_to_sym.values()) or not solver_interface.sol_statically_allocated:
            f.write('\n// User-defined variables\n')
            for name, value in variable_info.name_to_init.items():
                if variable_info.name_to_sym[name] or not solver_interface.sol_statically_allocated:
                    if not is_mathematical_scalar(value):
                        write_vec_prot(f, value.flatten(order='F'), f'{prefix}cpg_{name}', 'cpg_float')

        if not solver_interface.sol_statically_allocated:
            f.write('\n// Dual variables associated with user-defined constraints\n')
            for name, value in dual_variable_info.name_to_init.items():
                if not is_mathematical_scalar(value):
                    write_vec_prot(f, value, f'{prefix}cpg_{name}', 'cpg_float')

        f.write('\n// Struct containing primal solution\n')
        write_struct_prot(f, f'{prefix}CPG_Prim', 'CPG_Prim_t')

        if config.explicit != 1:
            if len(dual_variable_info.name_to_init) > 0:
                f.write('\n// Struct containing dual solution\n')
                write_struct_prot(f, f'{prefix}CPG_Dual', 'CPG_Dual_t')

        if not config.explicit:
            f.write('\n// Struct containing solver info\n')
            write_struct_prot(f, f'{prefix}CPG_Info', 'CPG_Info_t')

        f.write('\n// Struct containing solution and info\n')
        write_struct_prot(f, f'{prefix}CPG_Result', 'CPG_Result_t')

        if solver_interface.stgs_requires_extra_struct_type:
            f.write('\n// Struct containing solver settings\n')
            write_struct_prot(f, f'{prefix}Canon_Settings', 'Canon_Settings_t')

        if not solver_interface.ws_statically_allocated_in_solver_code:
            solver_interface.declare_workspace(f, prefix, parameter_canon)

        if config.gradient:

            f.write('\n// Derivative workspace\n')
            write_vec_prot(f, np.zeros(len(parameter_info.flat_usp)), f'{prefix}cpg_dp', 'cpg_float')

            f.write('\n// Struct containing parameter deltas\n')
            write_struct_prot(f, f'{prefix}CPG_Delta', 'CPG_Delta_t')


def write_solve_def(f, config, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface, parameter_canon_gradient=None):
    """
    Write parameter initialization function to file
    """

    write_description(f, 'c', 'Function definitions')
    f.write('#include "cpg_solve.h"\n')
    f.write('#include "cpg_workspace.h"\n')
    if config.gradient_two_stage:
        f.write('#include "cpg_gradient.h"\n')
        f.write('#include "cpg_gradient_workspace.h"\n')
    f.write('\n')

    f.write('static cpg_int i;\n')
    f.write('static cpg_int j;\n')
        
    if config.explicit:
        f.write('static cpg_float obj_val;\n')

    f.write('\n// Update user-defined parameters\n')
    for base_col, name in parameter_info.col_to_name_usp.items():
        Canon_outdated_names = parameter_canon.user_p_name_to_canon_outdated[name]
        if parameter_info.name_to_size_usp[name] == 1:
            f.write(f'void {config.prefix}cpg_update_{name}(cpg_float val){{\n')
            if config.explicit:
                f.write(f'  if (val < {config.prefix}cpg_params_vec_lower[{base_col}]){{\n')
                f.write(f'    val = {config.prefix}cpg_params_vec_lower[{base_col}];\n')
                f.write('  }\n')
                f.write(f'  if (val > {config.prefix}cpg_params_vec_upper[{base_col}]){{\n')
                f.write(f'    val = {config.prefix}cpg_params_vec_upper[{base_col}];\n')
                f.write('  }\n')
            f.write(f'  {config.prefix}cpg_params_vec[{base_col}] = val;\n')
        else:
            f.write(f'void {config.prefix}cpg_update_{name}(cpg_int idx, cpg_float val){{\n')
            if config.explicit:
                f.write(f'  if (val < {config.prefix}cpg_params_vec_lower[idx+{base_col}]){{\n')
                f.write(f'    val = {config.prefix}cpg_params_vec_lower[idx+{base_col}];\n')
                f.write('  }\n')
                f.write(f'  if (val > {config.prefix}cpg_params_vec_upper[idx+{base_col}]){{\n')
                f.write(f'    val = {config.prefix}cpg_params_vec_upper[idx+{base_col}];\n')
                f.write('  }\n')
            f.write(f'  {config.prefix}cpg_params_vec[idx+{base_col}] = val;\n')
        for Canon_outdated_name in Canon_outdated_names:
            f.write(f'  {config.prefix}Canon_Outdated.{Canon_outdated_name} = 1;\n')
            if config.gradient and not config.gradient_two_stage and Canon_outdated_name.isupper():
                f.write(f'  {config.prefix}Canon_Outdated_Grad.{Canon_outdated_name} = 1;\n')
        if config.gradient_two_stage:
            Canon_outdated_names_gradient = parameter_canon_gradient.user_p_name_to_canon_outdated[name]
            for Canon_outdated_name in Canon_outdated_names_gradient:
                f.write(f'  gradient_{config.prefix}Canon_Outdated_Grad.{Canon_outdated_name} = 1;\n')
        f.write('}\n\n')

    f.write('// Map user-defined to canonical parameters\n')

    for p_id, mapping in parameter_canon.p_id_to_mapping.items():
        if parameter_canon.p_id_to_changes[p_id]:
            f.write(f'void {config.prefix}cpg_canonicalize_{p_id}(){{\n')
            s = '->x' if p_id.isupper() else ''
            write_canonicalize(f, p_id, s, mapping, config.prefix)
            f.write('}\n\n')

    result_prefix = config.prefix if not solver_interface.ws_statically_allocated_in_solver_code else ''
    prim_str = result_prefix + solver_interface.ws_ptrs.primal_solution
    dual_str = result_prefix + solver_interface.ws_ptrs.dual_solution

    if solver_interface.ret_prim_func_exists(variable_info):
        f.write('// Retrieve primal solution in terms of user-defined variables\n')
        f.write(f'void {config.prefix}cpg_retrieve_prim(){{\n')
        if config.gradient_two_stage:
            f.write(f'  gradient_{config.prefix}cpg_retrieve_intermediate_primal();\n')
        for var_name, indices in variable_info.name_to_indices.items():
            if len(indices) == 1:
                f.write(f'  {config.prefix}CPG_Prim.{var_name} = {prim_str}[%d];\n' % indices[0])
            elif variable_info.name_to_sym[var_name] or not solver_interface.sol_statically_allocated:
                for i, idx in enumerate(indices):
                    f.write(f'  {config.prefix}CPG_Prim.{var_name}[%d] = {prim_str}[%d];\n' % (i, idx))
        f.write('}\n\n')

    if config.explicit != 1 and (solver_interface.ret_dual_func_exists(dual_variable_info) or config.gradient_two_stage):
        f.write('// Retrieve dual solution in terms of user-defined constraints\n')
        f.write(f'void {config.prefix}cpg_retrieve_dual(){{\n')
        if config.gradient_two_stage:
            # sum dual variable to sol_y
            f.write(f'  gradient_{config.prefix}cpg_retrieve_intermediate_dual();\n')
        for var_name, (canonical_var_name, indices) in dual_variable_info.name_to_indices.items():
            if len(indices) == 1:
                f.write(f'  {config.prefix}CPG_Dual.{var_name} = {dual_str.format(dual_var_name=canonical_var_name)}[%d];\n' % indices[0])
            elif not solver_interface.sol_statically_allocated:
                for i, idx in enumerate(indices):
                    f.write(f'  {config.prefix}CPG_Dual.{var_name}[%d] = {dual_str.format(dual_var_name=canonical_var_name)}[%d];\n' % (i, idx))
        f.write('}\n\n')

    if not config.explicit:
        f.write('// Retrieve solver info\n')
        f.write(f'void {config.prefix}cpg_retrieve_info(){{\n')
        f.write(f'  {config.prefix}CPG_Info.obj_val = {"-" if parameter_canon.is_maximization else ""}({result_prefix}{solver_interface.ws_ptrs.objective_value}{" + " + config.prefix + "Canon_Params.d" if parameter_canon.nonzero_d else ""});\n')
        f.write(f'  {config.prefix}CPG_Info.iter = {result_prefix}{solver_interface.ws_ptrs.iterations};\n')
        f.write(f'  {config.prefix}CPG_Info.status = {result_prefix}{solver_interface.ws_ptrs.status};\n')
        f.write(f'  {config.prefix}CPG_Info.pri_res = {result_prefix}{solver_interface.ws_ptrs.primal_residual};\n')
        f.write(f'  {config.prefix}CPG_Info.dua_res = {result_prefix}{solver_interface.ws_ptrs.dual_residual};\n')
        f.write('}\n\n')
    
    if solver_interface.inmemory_preconditioning:
        f.write('// Copy canonical parameters for preconditioning\n')
        for p_id, size in parameter_canon.p_id_to_size.items():
            if p_id != 'd':
                f.write(f'void {config.prefix}cpg_copy_{p_id}(){{\n')
                if size == 1:
                    f.write(f'  {config.prefix}Canon_Params_conditioning.{p_id} = {config.prefix}Canon_Params.{p_id};\n')
                elif size > 1:
                    f.write(f'  for (i=0; i<{size}; i++){{\n')
                    if p_id.isupper():
                        f.write(f'    {config.prefix}Canon_Params_conditioning.{p_id}->x[i] = {config.prefix}Canon_Params.{p_id}->x[i];\n')
                    else:
                        f.write(f'    {config.prefix}Canon_Params_conditioning.{p_id}[i] = {config.prefix}Canon_Params.{p_id}[i];\n')
                    f.write('  }\n')
                f.write('}\n\n')
        f.write(f'void {config.prefix}cpg_copy_all(){{\n')
        for p_id in parameter_canon.p.keys():
            if p_id != 'd':
                f.write(f'  {config.prefix}cpg_copy_{p_id}();\n')
        f.write('}\n\n')
        
    f.write('// Solve via canonicalization, canonical solve, retrieval\n')
    f.write(f'void {config.prefix}cpg_solve(){{\n')
    f.write('  // Canonicalize if necessary\n')

    for p_id, changes in parameter_canon.p_id_to_changes.items():
        if changes:
            f.write(f'  if ({config.prefix}Canon_Outdated.{p_id}) {{\n')
            f.write(f'    {config.prefix}cpg_canonicalize_{p_id}();\n')
            f.write('  }\n')

    if config.explicit:
        f.write('  // Update theta\n')
        k = 0
        for i, update in enumerate(parameter_canon.th_mask):
            if update:
                if i < solver_interface.n_var:
                    f.write(f'  {config.prefix}cpg_theta[{k}] = {config.prefix}Canon_Params.q[{i}];\n')
                else:
                    f.write(f'  {config.prefix}cpg_theta[{k}] = {config.prefix}Canon_Params.u[{i - solver_interface.n_var}];\n')
                k += 1
    else:
        pus = solver_interface.parameter_update_structure
        write_update_structure(f, config, parameter_canon, pus, *analyze_pus(pus, parameter_canon.p_id_to_changes))

    if solver_interface.stgs_dynamically_allocated:
        for name in solver_interface.stgs_names_to_type.keys():
            f.write(f'  {config.prefix}{solver_interface.ws_ptrs.settings.format(setting_name=name)} = {config.prefix}Canon_Settings.{name};\n')

    f.write(f'  // Solve with {config.solver_name}\n')
    f.write(f'  {solver_interface.solve_function_call.format(prefix=config.prefix)};\n')

    f.write('  // Retrieve results\n')
    if solver_interface.ret_prim_func_exists(variable_info):
        f.write(f'  {config.prefix}cpg_retrieve_prim();\n')
    if config.explicit != 1 and (solver_interface.ret_dual_func_exists(dual_variable_info) or config.gradient_two_stage):
        f.write(f'  {config.prefix}cpg_retrieve_dual();\n')
    if not config.explicit:
        f.write(f'  {config.prefix}cpg_retrieve_info();\n')

    f.write('  // Reset flags for outdated canonical parameters\n')
    for p_id, changes in parameter_canon.p_id_to_changes.items():
        if changes:
            f.write(f'  {config.prefix}Canon_Outdated.{p_id} = 0;\n')

    f.write('}\n\n')
    
    if config.explicit:
        if not variable_info.reduced:
            f.write(f'cpg_float {config.prefix}cpg_obj(){{\n')
            f.write(f'  obj_val = 0.0;\n')
            f.write(f'  for (i=0; i<{solver_interface.n_var}; i++){{\n')
            f.write(f'    for (j={config.prefix}Canon_Params.P->p[i]; j<{config.prefix}Canon_Params.P->p[i+1]; j++){{\n')
            f.write(f'      obj_val += 0.5 * {config.prefix}Canon_Params.P->x[j] * sol_x[{config.prefix}Canon_Params.P->i[j]] * sol_x[i];\n')
            f.write('    }\n')
            f.write('  }\n')
            f.write(f'  for (i=0; i<{solver_interface.n_var}; i++){{\n')
            f.write(f'    obj_val += {config.prefix}Canon_Params.q[i] * sol_x[i];\n')
            f.write('  }\n')
            f.write(f'  return obj_val;\n')
            f.write('}\n\n')

    if not config.explicit:  # TODO: explicit case
        f.write('// Update solver settings\n')
        f.write(f'void {config.prefix}cpg_set_solver_default_settings(){{\n')
        if solver_interface.stgs_reset_function is not None:
            f.write(f'  {solver_interface.stgs_reset_function["name"]}({solver_interface.stgs_reset_function["ptr"].format(prefix=config.prefix) if solver_interface.stgs_reset_function["ptr"] is not None else "&" + config.prefix + "Canon_Settings"});\n')
        else:
            for name, value in solver_interface.stgs_names_to_default.items():
                f.write(f'  {config.prefix}Canon_Settings.{name} = {value};\n')
        f.write('}\n')
        for name, typ in solver_interface.stgs_names_to_type.items():
            f.write(f'\nvoid {config.prefix}cpg_set_solver_{name}({typ} {name}_new){{\n')
            if solver_interface.stgs_direct_write_ptr is not None:
                f.write(f'  {solver_interface.stgs_direct_write_ptr.format(prefix=config.prefix)}->{name} = {name}_new;\n')
            else:
                f.write(f'  {config.prefix}Canon_Settings.{name} = {name}_new;\n')
            f.write('}\n')


def write_solve_prot(f, config, variable_info, dual_variable_info, parameter_info, parameter_canon, solver_interface, parameter_canon_gradient=None):
    """
    Write function declarations to file
    """

    write_description(f, 'c', 'Function declarations')
    f.write('#include "cpg_workspace.h"\n')
    if config.explicit:
        for header_file in solver_interface.header_files:
            f.write(f'#include {header_file}\n')

    f.write('\n// Update user-defined parameter values\n')
    for name, size in parameter_info.name_to_size_usp.items():
        if size == 1:
            f.write(f'extern void {config.prefix}cpg_update_{name}(cpg_float val);\n')
        else:
            f.write(f'extern void {config.prefix}cpg_update_{name}(cpg_int idx, cpg_float val);\n')

    f.write('\n// Map user-defined to canonical parameters\n')
    for p_id, changes in parameter_canon.p_id_to_changes.items():
        if changes:
            f.write(f'extern void {config.prefix}cpg_canonicalize_{p_id}();\n')

    if solver_interface.ret_prim_func_exists(variable_info):
        f.write('\n// Retrieve primal solution in terms of user-defined variables\n')
        f.write(f'extern void {config.prefix}cpg_retrieve_prim();\n')

    if config.explicit != 1 and solver_interface.ret_dual_func_exists(dual_variable_info):
        f.write('\n// Retrieve dual solution in terms of user-defined constraints\n')
        f.write(f'extern void {config.prefix}cpg_retrieve_dual();\n')

    if not config.explicit:
        f.write('\n// Retrieve solver information\n')
        f.write(f'extern void {config.prefix}cpg_retrieve_info();\n')

    if solver_interface.inmemory_preconditioning:
        f.write('\n// Copy canonical parameters for preconditioning\n')
        for p_id in parameter_canon.p_id_to_size.keys():
            if p_id != 'd':
                f.write(f'extern void {config.prefix}cpg_copy_{p_id}();\n')
        f.write(f'extern void {config.prefix}cpg_copy_all();\n')

    f.write('\n// Solve via canonicalization, canonical solve, retrieval\n')
    f.write(f'extern void {config.prefix}cpg_solve();\n')
    
    if config.explicit:
        if not variable_info.reduced:
            f.write('\n// Compute value of the objective\n')
            f.write(f'extern cpg_float {config.prefix}cpg_obj();\n')

    if not config.explicit:  # TODO: explicit case
        f.write('\n// Update solver settings\n')
        f.write(f'extern void {config.prefix}cpg_set_solver_default_settings();\n')
        for name, typ in solver_interface.stgs_names_to_type.items():
            f.write(f'extern void {config.prefix}cpg_set_solver_{name}({typ} {name}_new);\n')


def write_canon_cmake(f, name_parent_scope, solver_interface):
    """
    Pass sources to parent scope in <solver_code>/CMakeLists.txt
    """

    if len(solver_interface.cmake_headers) > 0:
        f.write('\nfile(\n  GLOB\n  SOLVER_HEAD')
        for h in solver_interface.cmake_headers:
            f.write('\n  ' + h)
        f.write(')\n')
        f.write(f'\nset({name_parent_scope}_head ${{SOLVER_HEAD}} PARENT_SCOPE)\n')
    if len(solver_interface.cmake_sources) > 0:
        f.write('\nfile(\n  GLOB\n  SOLVER_SRC')
        for h in solver_interface.cmake_sources:
            f.write('\n  ' + h)
        f.write(')')
        f.write(f'\nset({name_parent_scope}_src ${{SOLVER_SRC}} PARENT_SCOPE)\n')


def write_module_def(f, config, variable_info, dual_variable_info, parameter_info, solver_interface, gradient_interface):
    """
    Write c++ file for pbind11 wrapper
    """

    write_description(f, 'cpp', 'Python binding with pybind11')
    f.write('#include <pybind11/pybind11.h>\n')
    f.write('#include <pybind11/stl.h>\n')
    f.write('#include <ctime>\n')
    f.write('#include <chrono>\n')
    f.write('#include "cpg_module.hpp"\n\n')
    f.write('extern "C" {\n')
    f.write('    #include "include/cpg_workspace.h"\n')
    f.write('    #include "include/cpg_solve.h"\n')
    if config.gradient:
        f.write('    #include "include/cpg_gradient.h"\n')
        if not config.explicit:
            f.write('    #include "include/cpg_osqp_grad_workspace.h"\n')
            f.write('    #include "include/cpg_osqp_grad_compute.h"\n')
    if config.gradient_two_stage:
        f.write('    #include "include/cpg_gradient_workspace.h"\n')
    f.write('}\n\n')
    f.write('namespace py = pybind11;\n\n')
    if max(
            max(parameter_info.name_to_size_usp.values(), default=0),
            max(variable_info.name_to_size.values(), default=0),
            max(dual_variable_info.name_to_size.values(), default=0)
    ) > 1 or config.gradient:
        f.write('static int i;\n\n')

    # cpp function that maps parameters to results
    f.write(f'{config.prefix}CPG_Result_cpp_t {config.prefix}solve_cpp(struct {config.prefix}CPG_Updated_cpp_t& CPG_Updated_cpp, '
            f'struct {config.prefix}CPG_Params_cpp_t& CPG_Params_cpp){{\n\n')

    f.write('    // Pass changed user-defined parameter values to the solver\n')
    for name, size in parameter_info.name_to_size_usp.items():
        f.write(f'    if (CPG_Updated_cpp.{name}) {{\n')
        if size == 1:
            f.write(f'        {config.prefix}cpg_update_{name}(CPG_Params_cpp.{name});\n')
        else:
            f.write(f'        for(i=0; i<{size}; i++) {{\n')
            f.write(f'            {config.prefix}cpg_update_{name}(i, CPG_Params_cpp.{name}[i]);\n')
            f.write(f'        }}\n')
        f.write('    }\n')

    # perform ASA procedure
    f.write('\n    // Solve\n')
    f.write('    auto ASA_start = std::chrono::high_resolution_clock::now();\n')
    f.write(f'    {config.prefix}cpg_solve();\n')
    f.write('    auto ASA_end = std::chrono::high_resolution_clock::now();\n\n')

    # arrange and return results
    f.write('    // Arrange and return results\n')

    f.write(f'    {config.prefix}CPG_Prim_cpp_t CPG_Prim_cpp {{}};\n')
    for name, size in variable_info.name_to_size.items():
        if size == 1:
            f.write(f'    CPG_Prim_cpp.{name} = {config.prefix}CPG_Prim.{name};\n')
        else:
            if config.explicit:
                for i, idx in enumerate(variable_info.name_to_indices[name]):
                    if idx >= 0:
                        f.write(f'    CPG_Prim_cpp.{name}[{i}] = {config.prefix}CPG_Prim.{name}[{idx}];\n')
            else:
                f.write(f'    for(i=0; i<{size}; i++) {{\n')
                f.write(f'        CPG_Prim_cpp.{name}[i] = {config.prefix}CPG_Prim.{name}[i];\n')
                f.write('    }\n')

    if config.explicit != 1:
        if len(dual_variable_info.name_to_size) > 0:
            f.write(f'    {config.prefix}CPG_Dual_cpp_t CPG_Dual_cpp {{}};\n')
            for name, size in dual_variable_info.name_to_size.items():
                if size == 1:
                    f.write(f'    CPG_Dual_cpp.{name} = {config.prefix}CPG_Dual.{name};\n')
                else:
                    f.write(f'    for(i=0; i<{size}; i++) {{\n')
                    f.write(f'        CPG_Dual_cpp.{name}[i] = {config.prefix}CPG_Dual.{name}[i];\n')
                    f.write('    }\n')

    f.write(f'    {config.prefix}CPG_Info_cpp_t CPG_Info_cpp {{}};\n')
    if not config.explicit:  # TODO: explicit case
        for field in ['obj_val', 'iter', 'status', 'pri_res', 'dua_res']:
            f.write(f'    CPG_Info_cpp.{field} = {config.prefix}CPG_Info.{field};\n')
    f.write('    std::chrono::duration<double> elapsed = ASA_end - ASA_start;\n')
    f.write('    CPG_Info_cpp.time = elapsed.count();\n')
    #else:
    #    f.write('    CPG_Info_cpp.time = 1.0 * (ASA_end - ASA_start) / CLOCKS_PER_SEC / 1000;\n')
    if config.gradient:
        if config.explicit:
            f.write(f'    CPG_Info_cpp.region = pdaqp_active_region;\n')
        else:
            f.write(f'    for(i=0; i<{gradient_interface.n_var}; i++) {{\n')
            f.write(f'        CPG_Info_cpp.gradient_primal[i] = {f"gradient_{config.prefix}" if config.gradient_two_stage else ""}sol_x[i];\n')
            f.write('    }\n')
            f.write(f'    for(i=0; i<{gradient_interface.n_eq + gradient_interface.n_ineq}; i++) {{\n')
            f.write(f'        CPG_Info_cpp.gradient_dual[i] = {f"gradient_{config.prefix}" if config.gradient_two_stage else ""}sol_y[i];\n')
            f.write('    }\n')

    f.write(f'    {config.prefix}CPG_Result_cpp_t CPG_Result_cpp {{}};\n')
    f.write('    CPG_Result_cpp.prim = CPG_Prim_cpp;\n')
    if config.explicit != 1:
        if len(dual_variable_info.name_to_init) > 0:
            f.write('    CPG_Result_cpp.dual = CPG_Dual_cpp;\n')
    f.write('    CPG_Result_cpp.info = CPG_Info_cpp;\n')

    # return
    f.write('    return CPG_Result_cpp;\n\n')
    f.write('}\n\n')
    
    if config.gradient:
        # cpp function that maps variable deltas to parameter deltas
        if config.explicit:
            f.write(f'{config.prefix}CPG_PDelta_cpp_t {config.prefix}gradient_cpp('
                    f'struct {config.prefix}CPG_VDelta_cpp_t& CPG_VDelta_cpp, '
                    f'int region, bool use_sol){{\n\n')
        else:
            f.write(f'{config.prefix}CPG_PDelta_cpp_t {config.prefix}gradient_cpp('
                    f'struct {config.prefix}CPG_VDelta_cpp_t& CPG_VDelta_cpp, '
                    f'struct {config.prefix}CPG_GSol_cpp_t& CPG_GSol_cpp, bool use_sol){{\n\n')

        f.write('    // Set solution in canonical form for gradient computation\n')
        f.write('    if (use_sol) {\n')
        if config.explicit:
            f.write(f'        pdaqp_active_region = region;\n')
        else:
            f.write(f'        for(i=0; i<{gradient_interface.n_var}; i++) {{\n')
            f.write(f'            {f"gradient_{config.prefix}" if config.gradient_two_stage else ""}sol_x[i] = CPG_GSol_cpp.primal[i];\n')
            f.write('        }\n')
            f.write(f'        for(i=0; i<{gradient_interface.n_eq + gradient_interface.n_ineq}; i++) {{\n')
            f.write(f'            {f"gradient_{config.prefix}" if config.gradient_two_stage else ""}sol_y[i] = CPG_GSol_cpp.dual[i];\n')
            f.write('        }\n')
        f.write('    }\n\n')

        f.write('    // Pass user-defined variable deltas to the solver\n')
        for name, size in variable_info.name_to_size.items():
            if size == 1:
                f.write(f'    {config.prefix}cpg_update_d{name}(CPG_VDelta_cpp.{name});\n')
            else:
                if config.explicit:
                    for i, idx in enumerate(variable_info.name_to_indices[name]):
                        if idx >= 0:
                            f.write(f'    {config.prefix}cpg_update_d{name}({idx}, CPG_VDelta_cpp.{name}[{i}]);\n')
                else:
                    f.write(f'    for(i=0; i<{size}; i++) {{\n')
                    f.write(f'        {config.prefix}cpg_update_d{name}(i, CPG_VDelta_cpp.{name}[i]);\n')
                    f.write(f'    }}\n')

        # perform ASA gradient
        f.write('\n    // Compute gradient\n')
        f.write(f'    {config.prefix}cpg_gradient();\n\n')

        # arrange and return results
        f.write('    // Arrange and return results\n')

        f.write(f'    {config.prefix}CPG_PDelta_cpp_t CPG_PDelta_cpp {{}};\n')
        for name, size in parameter_info.name_to_size_usp.items():
            if size == 1:
                f.write(f'    CPG_PDelta_cpp.{name} = {config.prefix}CPG_Delta.{name}[0];\n')
            else:
                f.write(f'    for(i=0; i<{size}; i++) {{\n')
                f.write(f'        CPG_PDelta_cpp.{name}[i] = {config.prefix}CPG_Delta.{name}[i];\n')
                f.write('    }\n')

        # return
        f.write('    return CPG_PDelta_cpp;\n\n')
        f.write('}\n\n')

    # module
    f.write('PYBIND11_MODULE(cpg_module, m) {\n\n')

    f.write(f'    py::class_<{config.prefix}CPG_Params_cpp_t>(m, "{config.prefix}cpg_params")\n')
    f.write('            .def(py::init<>())\n')
    for name in parameter_info.name_to_size_usp.keys():
        f.write(f'            .def_readwrite("{name}", &{config.prefix}CPG_Params_cpp_t::{name})\n')
    f.write('            ;\n\n')

    f.write(f'    py::class_<{config.prefix}CPG_Updated_cpp_t>(m, "{config.prefix}cpg_updated")\n')
    f.write('            .def(py::init<>())\n')
    for name in parameter_info.name_to_size_usp.keys():
        f.write(f'            .def_readwrite("{name}", &{config.prefix}CPG_Updated_cpp_t::{name})\n')
    f.write('            ;\n\n')

    f.write(f'    py::class_<{config.prefix}CPG_Prim_cpp_t>(m, "{config.prefix}cpg_prim")\n')
    f.write('            .def(py::init<>())\n')
    for name in variable_info.name_to_init.keys():
        f.write(f'            .def_readwrite("{name}", &{config.prefix}CPG_Prim_cpp_t::{name})\n')
    f.write('            ;\n\n')

    if config.explicit != 1:
        if len(dual_variable_info.name_to_init) > 0:
            f.write(f'    py::class_<{config.prefix}CPG_Dual_cpp_t>(m, "{config.prefix}cpg_dual")\n')
            f.write('            .def(py::init<>())\n')
            for name in dual_variable_info.name_to_init.keys():
                f.write(f'            .def_readwrite("{name}", &{config.prefix}CPG_Dual_cpp_t::{name})\n')
            f.write('            ;\n\n')
        
    if config.gradient and not config.explicit:
        f.write(f'    py::class_<{config.prefix}CPG_GSol_cpp_t>(m, "{config.prefix}cpg_gsol")\n')
        f.write('            .def(py::init<>())\n')
        f.write(f'            .def_readwrite("primal", &{config.prefix}CPG_GSol_cpp_t::primal)\n')
        f.write(f'            .def_readwrite("dual", &{config.prefix}CPG_GSol_cpp_t::dual)\n')
        f.write('            ;\n\n')

    f.write(f'    py::class_<{config.prefix}CPG_Info_cpp_t>(m, "{config.prefix}cpg_info")\n')
    f.write('            .def(py::init<>())\n')
    if not config.explicit:  # TODO: explicit case
        f.write(f'            .def_readwrite("obj_val", &{config.prefix}CPG_Info_cpp_t::obj_val)\n')
        f.write(f'            .def_readwrite("iter", &{config.prefix}CPG_Info_cpp_t::iter)\n')
        f.write(f'            .def_readwrite("status", &{config.prefix}CPG_Info_cpp_t::status)\n')
        f.write(f'            .def_readwrite("pri_res", &{config.prefix}CPG_Info_cpp_t::pri_res)\n')
        f.write(f'            .def_readwrite("dua_res", &{config.prefix}CPG_Info_cpp_t::dua_res)\n')
    f.write(f'            .def_readwrite("time", &{config.prefix}CPG_Info_cpp_t::time)\n')
    if config.gradient:
        if config.explicit:
            f.write(f'            .def_readwrite("region", &{config.prefix}CPG_Info_cpp_t::region)\n')
        else:
            f.write(f'            .def_readwrite("gradient_primal", &{config.prefix}CPG_Info_cpp_t::gradient_primal)\n')
            f.write(f'            .def_readwrite("gradient_dual", &{config.prefix}CPG_Info_cpp_t::gradient_dual)\n')
    f.write('            ;\n\n')

    f.write(f'    py::class_<{config.prefix}CPG_Result_cpp_t>(m, "{config.prefix}cpg_result")\n')
    f.write('            .def(py::init<>())\n')
    f.write(f'            .def_readwrite("cpg_prim", &{config.prefix}CPG_Result_cpp_t::prim)\n')
    if config.explicit != 1:
        if len(dual_variable_info.name_to_init) > 0:
            f.write(f'            .def_readwrite("cpg_dual", &{config.prefix}CPG_Result_cpp_t::dual)\n')
    f.write(f'            .def_readwrite("cpg_info", &{config.prefix}CPG_Result_cpp_t::info)\n')
    f.write('            ;\n\n')

    f.write(f'    m.def("solve", &{config.prefix}solve_cpp);\n\n')

    if config.gradient:
        f.write(f'    py::class_<{config.prefix}CPG_VDelta_cpp_t>(m, "{config.prefix}cpg_vdelta")\n')
        f.write('            .def(py::init<>())\n')
        for name in variable_info.name_to_init.keys():
            f.write(f'            .def_readwrite("{name}", &{config.prefix}CPG_VDelta_cpp_t::{name})\n')
        f.write('            ;\n\n')
        
        f.write(f'    py::class_<{config.prefix}CPG_PDelta_cpp_t>(m, "{config.prefix}cpg_pdelta")\n')
        f.write('            .def(py::init<>())\n')
        for name in parameter_info.name_to_size_usp.keys():
            f.write(f'            .def_readwrite("{name}", &{config.prefix}CPG_PDelta_cpp_t::{name})\n')
        f.write('            ;\n\n')
        f.write(f'    m.def("gradient", &{config.prefix}gradient_cpp);\n\n')
    if not config.explicit:  # TODO: explicit case
        f.write(f'    m.def("set_solver_default_settings", &{config.prefix}cpg_set_solver_default_settings);\n')
        for name in solver_interface.stgs_names_to_type.keys():
            f.write(f'    m.def("set_solver_{name}", &{config.prefix}cpg_set_solver_{name});\n')

    f.write('\n}\n')


# ── Jinja2 template context builders ──────────────────────────────────────────


def _problem_summary_html(name_to_shape, name_to_size):
    """Return HTML table rows for a param/variable summary table."""
    rows = ''
    for n, sh in name_to_shape.items():
        if sh is None:
            shape_str = str(name_to_size[n])
        elif sh == ():
            shape_str = '1'
        elif len(sh) == 1:
            shape_str = str(sh[0])
        else:
            shape_str = f'{sh[0]} by {sh[1]} ({name_to_size[n]})'
        rows += f'      <tr>\n        <td><code>{n}</code></td>\n        <td>{shape_str}</td>\n      </tr>\n'
    return rows


def cmake_context(config):
    """Base context dict for CMakeLists.txt.jinja2.

    Solver-specific keys (extra_cmake_include_dirs, cmake_target_link_libs,
    cmake_definitions, solver_code_cmake_include_dir) are supplied by
    solver_interface.cmake_context_extra() and merged in writer.py.
    """
    return {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
        'prefix': config.prefix,
        'gradient': config.gradient,
        'explicit': config.explicit,
        'gradient_two_stage': config.gradient_two_stage,
    }



def readme_context(config, variable_info, dual_variable_info, parameter_info, solver_interface):
    """Context dict for README.html.jinja2."""

    prefix = config.prefix

    prim_typedef = '\n// Struct type with primal solution\ntypedef struct {\n'
    for name, size in variable_info.name_to_size.items():
        s = '' if size == 1 else '*'
        prim_typedef += f'  cpg_float    {(s + name + ";").ljust(9)}   // Your variable {name}\n'
    prim_typedef += '} CPG_Prim_t;\n'

    if dual_variable_info.name_to_size:
        dual_typedef = '\n// Struct type with dual solution\ntypedef struct {\n'
        for name, size in dual_variable_info.name_to_size.items():
            s = '' if size == 1 else '*'
            dual_typedef += f'  cpg_float    {(s + name + ";").ljust(9)}   // Your dual variable for constraint {name}\n'
        dual_typedef += '} CPG_Dual_t;\n\n'
    else:
        dual_typedef = ''

    info_typedef = '// Struct type with canonical solver information\ntypedef struct {\n'
    info_typedef += '  cpg_float    obj_val;    // Objective function value\n'
    info_typedef += '  cpg_int      iter;       // Number of iterations\n'
    info_typedef += (
        '  cpg_int      status;     // Solver status\n'
        if solver_interface.status_is_int else
        '  char         *status;    // Solver status\n'
    )
    info_typedef += '  cpg_float    pri_res;    // Primal residual\n'
    info_typedef += '  cpg_float    dua_res;    // Dual residual\n'
    info_typedef += '} CPG_Info_t;\n'

    result_typedef = '\n// Struct type with user-defined objective value and solution as fields\ntypedef struct {\n'
    result_typedef += '  CPG_Prim_t *prim;        // Primal solution\n'
    if dual_variable_info.name_to_init:
        result_typedef += '  CPG_Dual_t *dual;        // Dual solution\n'
    result_typedef += '  CPG_Info_t *info;        // Solver information\n'
    result_typedef += '} CPG_Result_t;\n'

    update_decls = '\n// Update user-defined parameter values\n'
    for name, size in parameter_info.name_to_size_usp.items():
        if size == 1:
            update_decls += f'void {prefix}cpg_update_{name}(cpg_float value);\n'
        else:
            update_decls += f'void {prefix}cpg_update_{name}(cpg_int idx, cpg_float value);\n'

    solve_decls = f'\n// Solve via canonicalization, canonical solve, retrieval\nvoid {prefix}cpg_solve();\n'

    settings_decls = (
        f'\n// Update solver settings\n'
        f'void {prefix}cpg_set_solver_default_settings();\n'
        f'void {prefix}cpg_set_solver_&lt;setting_name&gt;(&lt;setting_type&gt; &lt;setting_name&gt;_new);\n'
        '...\n'
    )

    settings_list = ', '.join([f'<code>{s}</code>' for s in solver_interface.stgs_names_enabled])

    return {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
        'params_table': _problem_summary_html(parameter_info.name_to_shape, parameter_info.name_to_size_usp),
        'primals_table': _problem_summary_html(variable_info.name_to_shape, variable_info.name_to_size),
        'duals_table': _problem_summary_html(dual_variable_info.name_to_shape, dual_variable_info.name_to_size),
        'code_dir': config.code_dir,
        'code_dir_python': config.code_dir.replace('/', '.').replace('\\', '.'),
        'solver_name': config.solver_name,
        'solver_docu_url': solver_interface.docu,
        'cmake_prefix': config.prefix + 'cpg',
        'gradient': config.gradient,
        'gradient_two_stage': config.gradient_two_stage,
        'prim_typedef': prim_typedef,
        'dual_typedef': dual_typedef,
        'info_typedef': info_typedef,
        'result_typedef': result_typedef,
        'update_declarations': update_decls,
        'solve_declarations': solve_decls,
        'settings_declarations': settings_decls,
        'settings_list': settings_list,
    }


def example_c_context(config, variable_info, dual_variable_info, parameter_info):
    """Context dict for cpg_example.c.jinja2."""
    writable_params = []
    for name, value in parameter_info.writable.items():
        if is_mathematical_scalar(value):
            writable_params.append((name, True, '%.20f' % squeeze_scalar(value)))
        else:
            writable_params.append((name, False, '%.20f' % value[0]))

    return {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
        'prefix': config.prefix,
        'gradient': config.gradient,
        'explicit': config.explicit,
        'writable_params': writable_params,
        'prim_name_to_size': list(variable_info.name_to_size.items()),
        'dual_name_to_size': list(dual_variable_info.name_to_size.items()),
        'param_name_to_size': list(parameter_info.name_to_size_usp.items()),
    }


def module_hpp_context(config, parameter_info, variable_info, dual_variable_info, solver_interface, gradient_interface):
    """Context dict for cpg_module.hpp.jinja2."""
    n_grad_var = gradient_interface.n_var if config.gradient else 0
    n_grad_constr = (gradient_interface.n_eq + gradient_interface.n_ineq) if config.gradient else 0
    return {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
        'prefix': config.prefix,
        'explicit': config.explicit,
        'gradient': config.gradient,
        'params': list(parameter_info.name_to_size_usp.items()),
        'param_names': list(parameter_info.name_to_size_usp.keys()),
        'prim_vars': list(variable_info.name_to_size.items()),
        'dual_vars': list(dual_variable_info.name_to_size.items()),
        'status_is_int': solver_interface.status_is_int,
        'n_grad_var': n_grad_var,
        'n_grad_constr': n_grad_constr,
    }


def solver_py_context(config, variable_info, dual_variable_info, parameter_info, solver_interface, gradient_interface):
    """Context dict for cpg_solver.py.jinja2."""
    param_names = list(parameter_info.name_to_size_usp.keys())
    param_names_quoted = ', '.join([f'"{name}"' for name in param_names])

    dual_save_list = []
    if config.explicit != 1:
        for name, parts in dual_variable_info.name_to_parts.items():
            constraint_idx = int(name[1:])  # d0 -> 0, d4 -> 4, etc.
            size = dual_variable_info.name_to_size[name]
            for j, local, shape in parts:
                index = '' if np.array_equal(local, np.arange(size)) else f'[{local.tolist()}]'
                dual_save_list.append((constraint_idx, j, name, index, shape))

    ctx = {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
        'prefix': config.prefix,
        'explicit': config.explicit,
        'gradient': config.gradient,
        'code_dir_module': config.code_dir.replace('/', '.').replace('\\', '.'),
        'stgs_translation': solver_interface.stgs_translation,
        'param_names_quoted': param_names_quoted,
        'param_names': param_names,
        'var_name_to_shape': list(variable_info.name_to_shape.items()),
        'dual_save_list': dual_save_list,
        'status_is_int': solver_interface.status_is_int,
        'status_int_fmt': (
            f'%d (for description visit {solver_interface.docu})'
            if solver_interface.status_is_int else ''
        ),
        'solver_name': config.solver_name,
        'special_settings': solver_interface.special_settings(config),
    }

    if config.gradient:
        ctx.update({
            'n_grad_var': gradient_interface.n_var,
            'n_grad_constr': gradient_interface.n_eq + gradient_interface.n_ineq,
            'prim_name_to_size': list(variable_info.name_to_size.items()),
            'param_name_to_shape': list(parameter_info.name_to_shape.items()),
        })

    return ctx


def grad_compute_context(config, gradient_interface):
    """Context dict for cpg_osqp_grad_compute.c.jinja2."""
    return {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
        'n': gradient_interface.n_var,
        'nl': gradient_interface.n_eq,
        'N': gradient_interface.n_var + gradient_interface.n_eq + gradient_interface.n_ineq,
        'workspace': f'{config.prefix}CPG_OSQP_Grad',
        'gradient_two_stage': config.gradient_two_stage,
        'sol_x_var': f'gradient_{config.prefix}sol_x' if config.gradient_two_stage else 'sol_x',
        'sol_y_var': f'gradient_{config.prefix}sol_y' if config.gradient_two_stage else 'sol_y',
    }
    
    
def grad_compute_h_context():
    """Context dict for cpg_osqp_grad_compute.h.jinja2."""
    return {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
    }


def grad_workspace_h_context(config):
    """Context dict for cpg_osqp_grad_workspace.h.jinja2."""
    return {
        'date': datetime.now().strftime("on %B %d, %Y at %H:%M:%S"),
        'workspace': f'{config.prefix}CPG_OSQP_Grad'
    }
