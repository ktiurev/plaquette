# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Convenient QEC simulations based on Clifford circuits."""
import typing as t
from collections.abc import Sequence

import numpy as np

import plaquette
from plaquette import circuit as plaq_circuit
from plaquette import device, pauli


class CircuitSimulator(device.AbstractSimulator):
    """QEC simulation based on Clifford circuits.

    This class provides a pure-Python tableau-based simulator which uses the
    representation of a stabilizer state introduced in :cite:`aaronson_improved_2004`.

    For a faster simulator based on a third-party package, see
    :class:`~plaquette.device._stimsim.StimSimulator`.

    .. automethod:: __init__
    """

    def __init__(self):
        """Create a new simulator."""
        #: State object
        self.state: None | pauli.Tableau = None
        self.circ: None | plaq_circuit.CircuitBuilder | plaq_circuit.Circuit = None
        """Circuit object to execute."""
        self.meas_results: list[int] = []
        """Measurement results collected while running the circuit"""
        self.erasure: list[bool] = []
        """Erasure information (one bool for each ``E_ERASE`` instruction)."""
        self.in_error: bool = False
        """Specifies whether the last instruction was an error instruction"""
        self.any_error_applied: bool = False
        """Specifies whether an error was already applied in the current error block"""
        self.apply_branch: bool = False
        """Specifies whether the current error branch is being applied."""

        self._instruction_pointer: int = -1
        """Index pointing to the **next** instruction to execute.

        Used internally to make the class usable as an iterator.

        .. warning:: Manually changing this should be done with extreme care.
        """

    def __iter__(self):
        """Iterate through instructions one-by-one.

        This allows you to run the circuit in a stepped fashion, enabling you
        to analyse/modify/play with the internal state of the
        simulator.

        .. important:: Each time a new iterator is made, the instruction
            pointer is reset, **but not the internal state**.
        """
        self._instruction_pointer = -1
        return self

    def __next__(self):
        """Step to the next gate/instruction in the circuit sequence.

        Notes:
            Each iteration step returns **nothing**, but rather updates the
            internal :attr:`.state` of the simulator and, if applicable, the
            :attr:`meas_results` and :attr:`erasure` attributes. You should
            inspect those if you want to do something based on measurement
            results.
        """
        self._instruction_pointer += 1
        if self._instruction_pointer < len(self.circ.gates):
            return self._run_gate(*self.circ.gates[self._instruction_pointer])
        raise StopIteration

    @property
    def n_qubits(self):
        """Number of qubits that this simulator handles.

        This depends on the circuit used to create the simulator.
        """
        return pauli.count_qubits(self.state)[0]

    def _handle_error(self, name, args):
        match name:
            case "ERROR":
                self.in_error = True
                if plaquette.rng.random() < args[0]:
                    self.any_error_applied = True
                    self.apply_branch = True
                    self._handle_gate(args[1], args[2:])
                else:
                    self.any_error_applied = False
                    self.apply_branch = False
            case "ERROR_CONTINUE":
                if not self.in_error:
                    raise ValueError("ERROR_CONTINUE not valid here")
                if self.apply_branch:
                    self._handle_gate(args[0], args[1:])
            case "ERROR_ELSE":
                if not self.in_error:
                    raise ValueError("ERROR_ELSE not valid here")
                if self.any_error_applied:
                    self.apply_branch = False
                elif plaquette.rng.random() < args[0]:
                    self.any_error_applied = True
                    self.apply_branch = True
                    self._handle_gate(args[1], args[2:])
            case _:
                raise ValueError(f"Unknown gate {name!r}")

    def _handle_gate(self, name: str, args: Sequence):
        if self.state is not None:
            match name:
                case "X":
                    self.state = pauli.x(self.state, args)
                case "Y":
                    self.state = pauli.y(self.state, args)
                case "Z":
                    self.state = pauli.z(self.state, args)
                case "H":
                    self.state = pauli.hadamard(self.state, args)
                case "CX":
                    self.state = pauli.cx(self.state, args[::2], args[1::2])
                case "CZ":
                    self.state = pauli.cz(self.state, args[::2], args[1::2])
                case "M":
                    n_q = self.n_qubits  # saves time in the loop
                    for q in args:
                        self.state, res = pauli.measure(
                            self.state,
                            pauli.single_qubit_pauli_operator("Z", q, n_q),
                        )
                        self.meas_results.append(res)
                case "R":
                    self.state = pauli.reset_qubits_to_eigenstate(self.state, "Z", args)
                case "DEPOLARIZE":
                    for qubit in args:
                        # Apply X, Y or Z with 33% probability each.
                        self._apply_pauli_from_int(
                            qubit, 1 + plaquette.rng.integers(0, 3)
                        )
                case "E_ERASE":
                    p, qubit = args
                    erase = plaquette.rng.random() < args[0]
                    self.erasure.append(erase)
                    if erase:
                        # Apply I, X, Y or Z with 25% probability each.
                        self._apply_pauli_from_int(qubit, plaquette.rng.integers(0, 4))
                case "E_PAULI":
                    p = (1 - sum(args[:3]), *args[:3])
                    qubit = args[3]
                    sample = plaquette.rng.choice(range(4), p=p)
                    self._apply_pauli_from_int(qubit, sample)
                case "E_PAULI2":
                    # Order of probabilities is:
                    # 0...15: II IX IY IZ XI XX ... ZY ZZ
                    p = (1 - sum(args[:15]), *args[:15])
                    qubits = args[15:]
                    sample = plaquette.rng.choice(range(16), p=p)
                    p1, p2 = divmod(sample, 4)
                    self._apply_pauli_from_int(qubits[0], p1)
                    self._apply_pauli_from_int(qubits[1], p2)
                case _:
                    raise ValueError(f"Unknown gate {name!r} (this should not happen)")

    def _apply_pauli_from_int(self, qubit: int, sample: int):
        """Apply Pauli ``"IXYZ"[sample]`` on ``qubit``."""
        if self.state is not None:
            match sample:
                case 0:
                    pass  # Do nothing
                case 1:
                    self.state = pauli.x(self.state, qubit)
                case 2:
                    self.state = pauli.y(self.state, qubit)
                case 3:
                    self.state = pauli.z(self.state, qubit)
                case _:
                    raise AssertionError("This should never happen")

    def _run_gate(self, name: str, args: Sequence[t.Any]):
        """Run a single gate."""
        if name.startswith("ERROR"):
            self._handle_error(name, args)
        else:
            self.in_error = False
            self._handle_gate(name, args)

    def run(
        self,
        circuit: plaq_circuit.Circuit | plaq_circuit.CircuitBuilder,
        *,
        shots=1,
        **kwargs,
    ):  # noqa: D102
        if isinstance(circuit, plaq_circuit.CircuitBuilder):
            self.circ = circuit.circ
        elif isinstance(circuit, plaq_circuit.Circuit):
            self.circ = circuit
        else:
            raise TypeError(
                "Only a Circuit or a CircuitBuilder can be used in a simulator"
            )

        if shots != 1:
            raise ValueError("The Stim simulator only allows running with shots=1.")

        after_reset = kwargs.get("after_reset", True)
        if after_reset:
            self.reset()

        self.state = pauli.zero_state(self.circ.number_of_qubits)

        for _ in self:
            # Go through all instructions
            pass

    def get_sample(  # noqa: D102
        self,
    ) -> tuple[np.ndarray, t.Optional[np.ndarray]]:
        if len(self.erasure) > 0:
            qubits_erased = np.array(self.erasure)
        else:
            qubits_erased = None
        return np.array(self.meas_results, dtype="u1"), qubits_erased

    def reset(self):
        """Reset the internal state of the simulator and its outputs.

        Notes:
            This will create a completely new :class:`.QuantumState` and clear
            the attributes :attr:`meas_results` and :attr:`erasure`.
        """
        self.state = None
        self.meas_results = []
        self.erasure = []
