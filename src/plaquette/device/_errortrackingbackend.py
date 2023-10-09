# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""QEC simulations based on tracking errors using Pauli frames."""
import typing as t
from collections.abc import Sequence

import numpy as np

import plaquette
from plaquette import circuit as plaq_circuit
from plaquette import pauli_frame as pauli_frame_ops
from plaquette.device._circuitsim import CircuitSimulator


class ErrorTrackingBackend:
    """Circuit simulator that tracks errors to provide samples.

    .. automethod:: __init__
    """

    def __init__(self):
        """Create a new error tracking circuit simulator.

        This simulator only runs the tableau simulation used by the
        CircuitSimulator once on the noiseless circuit to obtain a reference
        sample. Otherwise, it propagates a Pauli frame through the circuit to
        track errors. The backend returns a single sample by using the
        reference sample and a Pauli frame.
        """
        self.pauli_frame: None | np.ndarray = None
        """Specifies whether the last instruction was an error instruction."""

        self.circuit: None | plaq_circuit.Circuit = None
        """The circuit that is being run or was run last."""

        self.num_qubits: int = 0
        """The number of qubits from the circuit that is being run or was run last."""

        self.in_error: bool = False
        """Specifies whether the last instruction was an error instruction."""

        self.any_error_applied: bool = False
        """Specifies whether an error was already applied in the current error block."""

        self.apply_branch: bool = False
        """Specifies whether the current error branch is being applied."""

        self.meas_results: None | np.ndarray = None
        """The final noisy samples returned by the backend."""

        self.erasure: list[bool] = []
        """Erasure information (one bool for each ``E_ERASE`` instruction)."""

        self.ref_sample: None | np.ndarray = None
        """The sample erasure obtained from running the tableau simulation on
        the ideal circuit."""

        self.aux_simulator: None | CircuitSimulator = None
        """Simulator used to run the tableau simulation on the ideal circuit."""

    def reset(self):
        """Resets the backend."""
        self.pauli_frame = None
        self.circuit = None
        self.num_qubits = 0

        self.in_error = False
        self.any_error_applied = False
        self.apply_branch = False
        self.meas_results = None
        self.erasure = []
        self.ref_sample = None
        self.aux_simulator = None

    def run(
        self,
        circuit: plaq_circuit.Circuit | plaq_circuit.CircuitBuilder,
        *,
        shots=1,
    ):
        if isinstance(circuit, plaq_circuit.CircuitBuilder):
            circuit = circuit.circ
        elif not isinstance(circuit, plaq_circuit.Circuit):
            raise TypeError(
                "Only a Circuit or a CircuitBuilder can be used in a simulator"
            )

        # A new circuit was inputted - create a new reference sample
        if circuit is not self.circuit:
            self.circuit = circuit
            self.num_qubits = circuit.number_of_qubits

            self.ref_sample, _ = self.create_reference_sample(circuit)

        self.meas_results, self.erasure = self.perform_pauli_frame_simulation(circuit)

    def create_reference_sample(self, circuit: plaq_circuit.Circuit) -> np.ndarray:
        """Create a reference sample.

        The reference sample is created by running the tableau simulation on
        the noiseless circuit using the CircuitSimulator backend.

        Note that this method only returns samples and no information about:
        erasures because erasure simulation is  handled by the Pauli frame
        simulation step.

        Args:
            circuit: the circuit that is being run or was run last
        """
        no_noise_circuit = circuit.without_errors()

        # Create the reference sample
        self.aux_simulator = CircuitSimulator()
        self.aux_simulator.run(no_noise_circuit)

        return self.aux_simulator.get_sample()

    def perform_pauli_frame_simulation(
        self, circuit: plaq_circuit.Circuit
    ) -> tuple[np.ndarray, list]:
        """Performs the Pauli frame simulation to yield noisy samples.

        This method initializes a Pauli frame, propagates it through the
        circuit and produces measurement outcomes by using the previously
        generated reference sample for the circuit. It also keeps track of the
        outcome of erasure errors in the circuit.

        Args:
            circuit: the circuit that is being run or was run last
        """
        self.pauli_frame = pauli_frame_ops.init_pauli_frame(self.num_qubits)

        meas_results: list = []
        erasure: list = []

        for name, args in circuit.gates:
            if name == "M":
                if self.pauli_frame is not None and self.ref_sample is not None:
                    self.pauli_frame, sample = pauli_frame_ops.measure(
                        self.pauli_frame, self.ref_sample, args
                    )
                    meas_results.extend(sample)

            elif name == "E_ERASE":
                p, qubit = args
                if self.pauli_frame is not None:
                    self.pauli_frame, erased_or_not = pauli_frame_ops.erase(
                        self.pauli_frame, p, qubit
                    )
                    erasure.append(erased_or_not)
            else:
                self._handle_gate(name, args)

        return meas_results, erasure

    def get_sample(self) -> tuple[np.ndarray, t.Optional[np.ndarray]]:
        meas_results = np.array(self.meas_results, dtype=np.uint8)
        erasure = np.array(self.erasure) if self.erasure else None
        return meas_results, erasure

    ### Operation methods
    def _handle_gate(self, name: str, args: Sequence):
        match name:
            case "X":
                self.pauli_frame = pauli_frame_ops.x(self.pauli_frame, args)
            case "Y":
                self.pauli_frame = pauli_frame_ops.y(self.pauli_frame, args)
            case "Z":
                self.pauli_frame = pauli_frame_ops.z(self.pauli_frame, args)
            case "H":
                self.pauli_frame = pauli_frame_ops.hadamard(self.pauli_frame, args)
            case "CX":
                control_qubits, target_qubits = args[::2], args[1::2]
                self.pauli_frame = pauli_frame_ops.cx(
                    self.pauli_frame, control_qubits, target_qubits
                )
            case "CZ":
                control_qubits, target_qubits = args[::2], args[1::2]
                self.pauli_frame = pauli_frame_ops.cz(
                    self.pauli_frame, control_qubits, target_qubits
                )
            case "R":
                self.pauli_frame = pauli_frame_ops.reset_qubits(self.pauli_frame, args)
            case "DEPOLARIZE":
                self.pauli_frame = pauli_frame_ops.depolarize(self.pauli_frame, args)
            case "ERROR":
                p, error_type, *targets = args
                if error_type not in ("X", "Y", "Z"):
                    raise ValueError(f"ERROR ... {name!r} not supported yet")

                if len(targets) != 1:
                    raise ValueError("ERROR ... XYZ only supported on one qubit")

                self.in_error = True
                self.any_error_applied = self.apply_branch = plaquette.rng.random() < p

                if self.apply_branch:
                    error_type_int = pauli_frame_ops.error_op_int_mapping[error_type]
                    self.pauli_frame = pauli_frame_ops.apply_pauli_from_int_sample(
                        self.pauli_frame, error_type_int, targets
                    )
            case "ERROR_CONTINUE":
                if not self.in_error:
                    raise ValueError("ERROR_CONTINUE not valid here")

                error_type, *targets = args
                if error_type not in ("X", "Y", "Z"):
                    raise ValueError(f"ERROR ... {name!r} not supported yet")

                if len(targets) != 1:
                    raise ValueError("ERROR ... XYZ only supported on one qubit")

                if self.apply_branch:
                    error_type_int = pauli_frame_ops.error_op_int_mapping[error_type]
                    self.pauli_frame = pauli_frame_ops.apply_pauli_from_int_sample(
                        self.pauli_frame, error_type_int, targets
                    )
            case "ERROR_ELSE":
                p, error_type, *targets = args
                if not self.in_error:
                    raise ValueError("ERROR_ELSE not valid here")

                if error_type not in ("X", "Y", "Z"):
                    raise ValueError(f"ERROR ... {name!r} not supported yet")

                if len(targets) != 1:
                    raise ValueError("ERROR ... XYZ only supported on one qubit")

                if self.any_error_applied:
                    self.apply_branch = False
                elif plaquette.rng.random() < p:
                    self.any_error_applied = True
                    self.apply_branch = True
                    error_type_int = pauli_frame_ops.error_op_int_mapping[error_type]
                    self.pauli_frame = pauli_frame_ops.apply_pauli_from_int_sample(
                        self.pauli_frame, error_type_int, targets
                    )
            case "E_PAULI":
                self.pauli_frame = pauli_frame_ops.pauli_error_one_qubit(
                    self.pauli_frame, *args
                )
            case "E_PAULI2":
                self.pauli_frame = pauli_frame_ops.pauli_error_two_qubits(
                    self.pauli_frame, args
                )
            case _:
                raise ValueError(f"Unknown gate {name!r} (this should not happen)")
