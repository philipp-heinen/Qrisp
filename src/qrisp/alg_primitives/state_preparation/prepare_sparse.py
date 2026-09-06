"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import numpy as np
import scipy as sc
from qrisp import QuantumVariable, QuantumCircuit
from qrisp.circuit import U3Gate


def prepare_sparse(
    qv: QuantumVariable,
    target_state: sc.sparse.coo_array | dict,
    reverse: bool = False,
    method: str = "gleinig_hoefler",
):
    r"""Prepare a sparse quantum state on ``qv``. A quantum state is sparse if only a small fraction of all coefficients of that state, in the standard qubit basis, are non-zero. In this case, the circuit depth needed for the preparation of the state does not depend exponentially on the number of qubits any more, as for a generic state preparation algorithm, but only polynomially, while it also scales polynomially with the number S of non-zero coefficients.

    There are a number of sparse state preparation algorithms, optimizing different target quantities. For now, the function implements only the algorithm due to Gleinig and Hoefler (https://ieeexplore.ieee.org/abstract/document/9586240). It produces a circuit with O(S*n) CNOT gates and O(S*log(S)+n) single-qubit gates. The classical algorithm needed to find the circuit has a runtime of O(S^2*log(S)*n).

    Parameters
    ----------
    qv : QuantumVariable
        Quantum variable to prepare.
    target_state: scipy.sparse.coo_array or dict
        The sparse quantum state to prepare. It can either be provided as a sparse array of type scipy.sparse.coo_array or as a dictionary that contains the coefficient for every binary string. The binary strings (i.e. the dictionary keys) can be provided as str (e.g. ``'0111'``) or tuples (e.g. ``(0, 1, 1, 1)'``.
    reverse: bool
        Whether the standard little-endian convention should be reversed to a big-endian convention. This has only effect if ``target_state`` is provided as a sparse array instead of a dictionary. Default is False.
    method: str
        Which method to use. Currently only "gleinig_hoefler" is implemented, which employs the algorithm presented in https://ieeexplore.ieee.org/abstract/document/9586240.
    """
    if isinstance(target_state, sc.sparse.coo_array):
        if 2**len(qv) != target_state.shape[0]:
            raise ValueError(f"If target_state is array, it must have length 2**len(qv) but has length {target_state.shape[0]}")

        # convert the sparse vector to a dictionary with the coefficients
        indices = target_state.coords[0]
        values = target_state.data

        coeffs = {}
        for i, ind in enumerate(indices):
            coeffs[_int_to_binary_tuple(ind, len(qv), reverse)] = complex(values[i])

    elif isinstance(target_state, dict):
        if isinstance(list(target_state.keys())[0], tuple):
            if len(list(target_state.keys())[0]) != len(qv):
                raise ValueError(f"If target_state is a dictionary with tuples as keys, they must have size len(qv) but have size {len(list(target_state.keys())[0])}")

            coeffs = target_state

        elif isinstance(list(target_state.keys())[0], str):
            if len(list(target_state.keys())[0]) != len(qv):
                raise ValueError(f"If target_state is a dictionary with strings as keys, they must have size len(qv) but have size {len(list(target_state.keys())[0])}")

            coeffs = {tuple(map(int, k)): v for k, v in target_state.items()}   # convert strings to tuples for later convenience

        else:
            raise ValueError(f"If target_state is a dictionary, its keys must be either tuples or strings, but are of type {type(list(target_state.keys())[0])}")

    else:
        raise ValueError(f"target_state must be either of type scipy.sparse.coo_array or dict, but is of type {type(target_state)}")

    if method == "gleinig_hoefler":
        _prepare_gleinig_hoefler(qv, coeffs)
    else:
        raise ValueError(f"Method {method} not known.")


def _prepare_gleinig_hoefler(qv: QuantumVariable, coeffs: dict):
    r"""Prepare a sparse quantum state according to the Gleinig-Hoefler algorithm. It computes a circuit that iteratively reduces the number of non-zero entries in the target state (by calling _gleinig_hoefler_subroutine) until only one single state remains, which is trivial to transform into |000...>. By reversing the resulting circuit, the target state can be prepared from |000...>.
    """
    n = len(qv)

    S =  list(coeffs.keys())

    qc = QuantumCircuit(n)

    while len(S) > 1:
        S, coeffs = _gleinig_hoefler_subroutine(qc, S, coeffs, n)

    for i, b in enumerate(S[0]):
        if b:
            qc.x(i)

    qc.gphase(-np.angle(list(coeffs.values())[0]), 0)

    qv.append(qc.inverse().to_op())


def _gleinig_hoefler_subroutine(qc: QuantumCircuit, S: list[tuple], coeffs: dict, n: int):
    r"""
    Subroutine of the Gleinig-Hoefler algorithm, which reduces the number of non-zero entries in the state in every step.
    """
    diff_qubits = []
    diff_values = []

    def reduction_loop(T):
        while len(T) > 1:
            max_diff = -1
            max_i = 0
            for i in range(n):
                _T0 = [s for s in T if s[i] == 0]
                _T1 = [s for s in T if s[i] == 1]
                if len(_T0) != 0 and len(_T1) != 0 and abs(len(_T0) - len(_T1)) > max_diff:
                    max_i = i
                    max_diff = abs(len(_T0) - len(_T1))

            diff_qubits.append(max_i)
            T0 = [s for s in T if s[max_i] == 0]
            T1 = [s for s in T if s[max_i] == 1]

            if len(T0) < len(T1):
                T = T0
                diff_values.append(0)
            else:
                T = T1
                diff_values.append(1)
        return T

    T = reduction_loop(S)

    diff = diff_qubits.pop(-1)
    diff_values.pop(-1)
    x1 = T[0]

    U = [s for s in S if all([s[j] == diff_values[i] for i, j in enumerate(diff_qubits)])]
    U.remove(x1)
    U = reduction_loop(U)
    x2 = U[0]


    coeffs_new = coeffs.copy()
    x1_new = list(x1)
    x2_new = list(x2)

    if x1[diff] != 1:
        qc.x(diff)

        coeffs_new = {(*s[:diff], s[diff] ^ 1, *s[diff+1:]): coeff for s, coeff in coeffs_new.items()}
        x1_new[diff] = x1_new[diff] ^ 1
        x2_new[diff] = x2_new[diff] ^ 1

    for i in list(range(n)):
        if i != diff and x1[i] != x2[i]:
            qc.cx(diff, i)

            coeffs_new = {(*s[:i], s[i] ^ s[diff], *s[i+1:]): coeff for s, coeff in coeffs_new.items()}
            x1_new[i] = x1_new[i] ^ x1_new[diff]
            x2_new[i] = x2_new[i] ^ x2_new[diff]

    for i in diff_qubits:
        if x2[i] != 1:
            qc.x(i)

            coeffs_new = {(*s[:i], s[i] ^ 1, *s[i+1:]): coeff for s, coeff in coeffs_new.items()}
            x1_new[i] = x1_new[i] ^ 1
            x2_new[i] = x2_new[i] ^ 1

    alpha, beta = coeffs[x1], coeffs[x2]

    newvalue = _rotate_state_to_0(qc, diff, diff_qubits, beta, alpha)

    coeffs_new.pop(tuple(x1_new))
    coeffs_new[tuple(x2_new)] = newvalue

    return list(coeffs_new.keys()), coeffs_new


def _int_to_binary_tuple(n: int, width: int, reverse: bool):
    r"""
    Convert an integer to a binary tuple.
    """
    bits = []
    while n > 0:
        bits.append(int(n & 1))
        n >>= 1
    bits.extend([0] * (width - len(bits)))
    if reverse:
        return tuple(bits)
    else:
        return tuple(reversed(bits))


def _normal_form(alpha: complex | float, beta: complex | float):
    r"""
    Bring a state alpha |0> + beta |1> to normal form, i.e. normalize and make the phase of alpha 0.
    """
    tot_phase = np.angle(alpha)
    alpha *= np.exp(-1j*tot_phase)
    beta *= np.exp(-1j*tot_phase)

    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)

    return alpha / norm, beta / norm, tot_phase, norm


def _rotate_state_to_0(qc: QuantumCircuit, target: int, control: list, alpha: complex | float, beta: complex | float):
    r"""
    Rotate a state alpha |0> + beta |1> in qubit ``target`` to the state |0>, controlled by the qubits in ``control``.
    """
    _alpha, _beta, tot_phase, norm = _normal_form(alpha, beta)

    theta = 2*np.acos(np.abs(_alpha))
    phi = np.angle(_beta)

    if len(control) > 0:
        gate = U3Gate(-theta, 0, -phi).control(len(control))
    else:
        gate = U3Gate(-theta, 0, -phi)

    qc.append(gate, qubits=control+[target])

    return np.exp(1j*tot_phase)*norm
