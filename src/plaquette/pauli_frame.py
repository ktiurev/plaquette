# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Definition of a Pauli frame and effects of quantum operations on frames.

A Pauli frame is a Pauli operator that is tracking the Pauli errors that have
occurred on each qubit in our system. This module implements how quantum
operations of stabilizer quantum error-correcting codes act on the Pauli frame
data structure.
"""
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

import plaquette

PauliFrame: TypeAlias = np.ndarray[np.dtype[np.uint8]]
error_op_int_mapping = {"X": 1, "Y": 2, "Z": 3}


def validate_frame_length(pauli_frame: PauliFrame):
    """Validate the length of the Pauli frame.

    The frame should be twice the length of the number of qubits.

    Args:
        pauli_frame: the Pauli frame to update
    """
    if len(pauli_frame) % 2 != 0:
        raise ValueError(
            (
                "The Pauli frame should have a size twice "
                "the number of qubits it represents."
            )
        )


def get_num_qubits_from_frame(pauli_frame: PauliFrame) -> int:
    """Calculates the number of qubits the Pauli frame is representing.

    Args:
        pauli_frame: the Pauli frame to update
    """
    validate_frame_length(pauli_frame)
    return len(pauli_frame) // 2


def maybe_apply_z(pauli_frame: PauliFrame, qubit: int) -> PauliFrame:
    """Apply a Pauli Z operator to a qubit with a 50% probability.

    This method updates the Pauli frame and is usually called when a
    collapsing operation was applied to the quantum state.

    Args:
        pauli_frame: the Pauli frame to update
        qubit: the qubit index
    """
    num_qubits = get_num_qubits_from_frame(pauli_frame)
    if plaquette.rng.integers(2):
        target = num_qubits + qubit
        pauli_frame[target] ^= 1
    return pauli_frame


def init_pauli_frame(num_qubits: int) -> PauliFrame:
    """Initializes a Pauli frame for the specified number of qubits.

    A Pauli Z error is applied to each qubit at random due to the
    initialization step.

    Args:
        num_qubits: the number of qubits to create the frame for
    """
    pauli_frame = np.zeros(2 * num_qubits, dtype=np.uint8)

    for i in range(num_qubits):
        pauli_frame = maybe_apply_z(pauli_frame, i)

    return pauli_frame


def apply_pauli_from_int_sample(
    pauli_frame: PauliFrame, sample: int, qubits: int | Sequence
) -> PauliFrame:
    """Applies a Pauli operator based on an integer sample to a single qubit.

    A Pauli is applied if the integer sample is between 0 and 3:

    * 0: Identity is applied (frame is returned unchanged)
    * 1: PauliX is applied
    * 2: PauliY is applied
    * 3: PauliZ is applied

    Args:
        pauli_frame: the Pauli frame to update
        sample: the integer representation of the Pauli operator on a single
            qubit
        qubits: the qubit or qubits to apply the operators to
    """
    if not isinstance(sample, (int, np.integer)) or sample < 0 or sample > 3:
        raise ValueError("The input integer has to be 0, 1, 2 or 3.")

    num_qubits = get_num_qubits_from_frame(pauli_frame)

    apply_x = sample == 1
    apply_y = sample == 2
    apply_z = sample == 3

    if not isinstance(qubits, Sequence):
        qubits = [qubits]

    for q in qubits:
        if apply_x or apply_y:
            # Record a Pauli X error
            pauli_frame[q] ^= 1

        if apply_y or apply_z:
            # Record a Pauli Z error
            pauli_frame[num_qubits + q] ^= 1

    return pauli_frame


def pauli_error_one_qubit(
    pauli_frame: PauliFrame, x_prob: float, y_prob: float, z_prob: float, qubit: int
) -> PauliFrame:
    """Randomly apply a Pauli operator on the specified qubit with some probabilities.

    Args:
        pauli_frame: the Pauli frame to update
        x_prob: probability of applying a Pauli X operator
        y_prob: probability of applying a Pauli Y operator
        z_prob: probability of applying a Pauli Z operator
        qubit: target qubit
    """
    p = (1 - sum([x_prob, y_prob, z_prob]), x_prob, y_prob, z_prob)
    sample = plaquette.rng.choice(range(4), p=p)
    return apply_pauli_from_int_sample(pauli_frame, sample, qubit)


def pauli_error_two_qubits(pauli_frame: PauliFrame, args: Sequence) -> PauliFrame:
    """Apply Pauli ``"IXYZ"[sample]`` on ``qubit``.

    The expected order of the probabilities is:

    0...15: II IX IY IZ XI XX ... ZY ZZ

    Args:
        pauli_frame: the Pauli frame to update
        args: the probabilities and the qubit indices
    """
    p = (1 - sum(args[:15]), *args[:15])
    qubits = args[15:]
    sample = plaquette.rng.choice(range(16), p=p)
    p1, p2 = divmod(sample, 4)
    pauli_frame = apply_pauli_from_int_sample(pauli_frame, p1, qubits[0])
    pauli_frame = apply_pauli_from_int_sample(pauli_frame, p2, qubits[1])
    return pauli_frame


def erase(pauli_frame: PauliFrame, p: float, qubit: int) -> tuple[PauliFrame, bool]:
    r"""Erase the target qubit.

    Args:
        pauli_frame: the Pauli frame to update
        p: probability of erasing the qubit
        qubit: target qubit
    """
    qubit_erased = plaquette.rng.random() < p
    if qubit_erased:
        # The qubit was erased, so we update the Pauli frame
        pauli_frame = pauli_error_one_qubit(pauli_frame, 0.25, 0.25, 0.25, qubit)
    return pauli_frame, qubit_erased


def depolarize(pauli_frame: PauliFrame, targets: Sequence[int]) -> PauliFrame:
    r"""Apply depolarization to each of the target qubits.

    Applies an X, Y or Z Pauli error with 33% probability each.

    Args:
        pauli_frame: the Pauli frame to update
        targets: target qubits
    """
    for qubit in targets:
        pauli_error_int = 1 + plaquette.rng.integers(0, 3)
        pauli_frame = apply_pauli_from_int_sample(pauli_frame, pauli_error_int, qubit)
    return pauli_frame


def hadamard(pauli_frame: PauliFrame, targets: Sequence[int]) -> PauliFrame:
    r"""Conjugate the frame with a Hadamard on the target qubits.

    H transforms stabilizers according to :math:`Z \mapsto X` and
    :math:`X \mapsto Z`

    Args:
        pauli_frame: the Pauli frame to update
        targets: 0-based target qubit index/indices.
    """
    num_qubits = get_num_qubits_from_frame(pauli_frame)
    x_comp_idx = np.array(targets)
    z_comp_idx = np.array(targets) + num_qubits

    s = pauli_frame[x_comp_idx]
    pauli_frame[x_comp_idx] = pauli_frame[z_comp_idx]
    pauli_frame[z_comp_idx] = s
    return pauli_frame


def x(pauli_frame: PauliFrame, targets: Sequence[int]) -> PauliFrame:
    """Conjugate the frame with a Pauli X on the target qubits.

    This operation leaves the records in the frame unchanged:
      * I: XIX = I
      * X: XXX = X
      * Y: XYX = -Y
      * Z: XZX = -Z

    Args:
        pauli_frame: the Pauli frame to update
        targets: 0-based target qubit index/indices.
    """
    return pauli_frame


def y(pauli_frame: PauliFrame, targets: Sequence[int]) -> PauliFrame:
    """Conjugate the frame with a Pauli Y on the target qubits.

    This operation leaves the records in the frame unchanged:
      * I: YIY = I
      * X: YXY = -X
      * Y: YYY = Y
      * Z: YZY = -Z

    Args:
        pauli_frame: the Pauli frame to update
        targets: 0-based target qubit index/indices.
    """
    return pauli_frame


def z(pauli_frame: PauliFrame, targets: Sequence[int]) -> PauliFrame:
    """Conjugate the frame with a Pauli Z on the target qubits.

    This operation leaves the records in the frame unchanged:
      * I: ZIZ = I
      * X: ZXZ = -X
      * Y: ZYZ = -Y
      * Z: ZZZ = Z

    Args:
        pauli_frame: the Pauli frame to update
        targets: 0-based target qubit index/indices.
    """
    return pauli_frame


def controlled_not_gate(
    pauli_frame: PauliFrame, control_qubits: Sequence[int], target_qubits: Sequence[int]
) -> PauliFrame:
    """Perform the CNOT gate on the specified qubits.

    Args:
        pauli_frame: the Pauli frame to update
        control_qubits: The control qubits.
        target_qubits: The target qubits.
    """
    if len(control_qubits) != len(target_qubits):
        msg = (
            "The number of control qubits "
            "should be equal to the number of target qubits."
        )
        raise ValueError(msg)

    num_qubits = get_num_qubits_from_frame(pauli_frame)

    num_cnots = int(len(control_qubits))
    for idx in range(num_cnots):
        control = control_qubits[idx]
        target = target_qubits[idx]

        # Apply X rule
        if pauli_frame[control]:
            pauli_frame[target] ^= 1

        # Apply Z rule
        if pauli_frame[target + num_qubits]:
            pauli_frame[control + num_qubits] ^= 1
    return pauli_frame


cx = controlled_not_gate


def controlled_phase_gate(
    pauli_frame: PauliFrame, control_qubits: Sequence[int], target_qubits: Sequence[int]
) -> PauliFrame:
    """Perform the CZ gate on the specified qubits.

    Args:
        pauli_frame: the Pauli frame to update
        control_qubits: The control qubits.
        target_qubits: The target qubits.
    """
    if len(control_qubits) != len(target_qubits):
        msg = (
            "The number of control qubits "
            "should be equal to the number of target qubits."
        )
        raise ValueError(msg)

    num_qubits = get_num_qubits_from_frame(pauli_frame)

    if pauli_frame is not None:
        num_cnots = int(len(control_qubits))
        for idx in range(num_cnots):
            control = control_qubits[idx]
            target = target_qubits[idx]

            # Apply X rules
            if pauli_frame[control]:
                pauli_frame[target + num_qubits] ^= 1

            if pauli_frame[target]:
                pauli_frame[control + num_qubits] ^= 1
    return pauli_frame


cz = controlled_phase_gate


def reset_qubits(pauli_frame: PauliFrame, targets: Sequence[int]) -> PauliFrame:
    """Apply qubit reset operation on ``targets``.

    Args:
        pauli_frame: the Pauli frame to update
        targets: the target qubits
    """
    num_qubits = get_num_qubits_from_frame(pauli_frame)

    x_comp_idx = np.array(targets)
    z_comp_idx = np.array(targets) + num_qubits

    # Set the qubit's term in the frame to identity
    pauli_frame[x_comp_idx] = 0
    pauli_frame[z_comp_idx] = 0

    for target in targets:
        pauli_frame = maybe_apply_z(pauli_frame, target)
    return pauli_frame


def measure(
    pauli_frame: PauliFrame, ref_sample: np.ndarray, qubits: Sequence[int]
) -> tuple[PauliFrame, np.ndarray]:
    """Measure the target qubits by using the reference sample and the Pauli frame.

    The measurement outcome (final sample) of a qubit is obtained by
    performing an exclusive or (XOR) operation between the reference sample
    of the qubit and the X error component of the qubit in the Pauli frame.

    After measurements, a Pauli Z operation is applied to the measured
    qubits as measurements are collapsing operations.

    Args:
        pauli_frame: the Pauli frame to update
        ref_sample: the reference sample
        qubits: the qubits to measure
    """
    target_x_indices = list(qubits)

    # r_M XOR x_q
    measurement_outcomes = ref_sample[target_x_indices] ^ pauli_frame[target_x_indices]
    for qubit in qubits:
        pauli_frame = maybe_apply_z(pauli_frame, qubit)
    return pauli_frame, measurement_outcomes
