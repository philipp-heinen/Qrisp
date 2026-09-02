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
from qrisp import QuantumVariable, QuantumCircuit
from qrisp.circuit import U3Gate


def prepare_sparse(
    qv: QuantumVariable,
    target_array,
    reversed: bool = False,
    method: str = "gleinig_hoefler",
):
    if method == "gleinig_hoefler":
        _prepare_gleinig_hoefler(qv, target_array, reversed)
    else:
        raise ValueError(f"Method {method} not known.")


def _prepare_gleinig_hoefler(
    qv: QuantumVariable,
    target_array,
    reversed: bool = False,
):
    
    pass

def _gleinig_hoefler_subroutine(non_zero_indices: list[tuple], non_zero_values: list[complex]):
    diff_qubits = []
    diff_values = []
    
    T = non_zero_indices
    
def _normal_form(alpha: complex | float, beta: complex | float):
    tot_phase = np.angle(alpha)
    alpha *= np.exp(-1j*tot_phase)
    beta *= np.exp(-1j*tot_phase)
    
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    
    return alpha / norm, beta / norm

def _rotate_state_to_0(qc: QuantumCircuit, target: int, control: list, alpha: complex | float, beta: complex | float):
    alpha, beta = _normal_form(alpha, beta)
        
    theta = 2*np.acos(np.abs(alpha))
    phi = np.angle(beta)    
    
    gate = U3Gate(-theta, 0, -phi).control(len(control))
    qc.append(gate, qubits=[target]+control)
