# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Quantum devices and error correction simulators.

This module provides tools for connecting to quantum devices and simulating
quantum error correction using suitable Clifford circuits. A circuit from
:mod:`plaquette.circuit` can be simulated as follows:

>>> from plaquette.codes import LatticeCode
>>> from plaquette.circuit.generator import generate_qec_circuit
>>> from plaquette import Device
>>> circ = generate_qec_circuit(LatticeCode.make_planar(1, 4), {}, {}, logical_ops="X")
>>> dev = Device("clifford")
>>> dev  # doctest: +ELLIPSIS
<plaquette.device.Device object at ...>

In addition to the built-in pure-Python circuit simulator backend, the faster
Stim simulator can be used by specifying ``"stim"`` as the backend:

>>> dev = Device("stim")
>>> dev.run(circ)
>>> raw, erasure = dev.get_sample()

``raw`` contains all the measurement results from measurement gates in the
circuit, while ``erasure`` contains information on erased qubits if
the :ref:`Gate E_ERASE` was used. The circuit returns measurement results as a
linear array and the function
:meth:`.MeasurementSample.from_code_and_raw_results` can be used to split this
array into different parts (this assumes that the circuit was generated using
:mod:`plaquette.circuit`).
"""

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pkg_resources

from plaquette import circuit as plaq_circuit
from plaquette.codes import LatticeCode
from plaquette.pauli import count_qubits


@dataclass
class MeasurementSample:
    """One sample from a simulator.

    .. automethod:: __init__
    """

    logical_op_initial: np.ndarray
    """Measurement results for logical operators before QEC.

    Array with one entry for each logical operator (0 or 1).
    """
    logical_op_final: np.ndarray
    """Measurement results for logical operators after QEC.
    Array with one entry for each logical operator (0 or 1).
    """
    logical_op_toggle: np.ndarray
    """XOR between logical operator values before and after QEC.

    Array with one entry for each logical operator (0 or 1).
    """
    stabilizer_gen: np.ndarray
    """Measurement results for stabilizer generators.

    The shape is ``(n_rounds + 1, n_stabgens)``. The ``+1`` is necessary because it
    always includes one round of "initial" stabilizer measurements to prepare the
    initial state.
    """
    syndrome: np.ndarray
    """Syndrome data (derived from measurement results for stabilizer generators).

    The shape is ``(n_rounds, n_stabgens)``. It is obtained from ``stabilizer_gen``
    by taking the XOR of consecutive rounds.

    For details of the relation between measurement results and syndrome for multiple
    rounds of stabilizer measurements, see :class:`plaquette.syngraph.Vertex`.
    """
    erased_qubits: Optional[np.ndarray]
    """Erasure information

    The shape is ``(n_rounds, n_qubits)`` (only data qubits, no ancilla qubits).
    Each entry is a boolean and specifies whether the given qubit was erased in the
    given round.
    """

    @classmethod
    def from_code_and_raw_results(
        cls,
        code: LatticeCode,
        raw_results: np.ndarray,
        erasure: Optional[np.ndarray] = None,
        logical_ancilla: bool = False,
    ):
        """Unpack the results from a simulator into a more convenient format.

        Args:
            code: the :class:`.LatticeCode` used to generate the circuit which produced
                the results you want to unpack.
            raw_results: the measurement results from
                :meth:`AbstractSimulator.get_sample`.
            erasure: erasure information from
                :meth:`AbstractSimulator.get_sample`.
            logical_ancilla: flag for alternative method to measure logical operators
                via additional ancilla that is entangled with several physical qubits
        """
        if not logical_ancilla:
            # Default behavior
            code_distance = count_qubits(code.logical_ops, include_identities=False)[0]

            logical_op_initial = np.array(
                [
                    np.sum(raw_results[i * code_distance : (i + 1) * code_distance]) % 2
                    for i in range(code.n_logical_qubits)
                ],
                dtype=int,
            )
            # fmt: off
            logical_op_final = np.array(
                [
                    np.sum(
                        raw_results[len(raw_results) - (i * code_distance)
                            - code_distance : len(raw_results) - (i * code_distance)]  # noqa
                    ) % 2 for i in range(code.n_logical_qubits)
                ][::-1], dtype=int,
            )
            ancillas = np.array(
                raw_results[code.n_logical_qubits * code_distance : -code.n_logical_qubits * code_distance] # noqa
            )
            # fmt: on
        else:
            # Alternative behavior
            logical_op_initial = np.array(raw_results[: code.n_logical_qubits])
            logical_op_final = np.array(raw_results[-code.n_logical_qubits :])
            ancillas = np.array(
                raw_results[code.n_logical_qubits : -code.n_logical_qubits]
            )

        logical_toggle = logical_op_initial ^ logical_op_final
        stab = ancillas.reshape((code.n_rounds + 1, code.n_stabgens))
        syndrome = stab[1:] ^ stab[:-1]

        if erasure is not None:
            if erasure.shape != (code.n_rounds * code.n_data_qubits,):
                raise ValueError("Wrong number of erasure information pieces")
            erasure = erasure.reshape((code.n_rounds, code.n_data_qubits))
        else:
            erasure = None

        return MeasurementSample(
            logical_op_initial=logical_op_initial,
            logical_op_final=logical_op_final,
            logical_op_toggle=logical_toggle,
            stabilizer_gen=stab,
            syndrome=syndrome,
            erased_qubits=erasure,
        )


local_simulators = {"clifford", "stim", "tableau"}


# The recognized quantum devices.
# Note that these are loaded once when the module has been loaded.
recognized_devices = {
    entry.name: entry for entry in pkg_resources.iter_entry_points("plaquette.device")
}


class Device:
    """Quantum device for accessing simulators or real quantum hardware.

    .. automethod:: __init__
    """

    def __init__(self, backend: str, *args, **kwargs):
        """Create a new quantum device.

        There are two built-in backends: ``"clifford"``
        (:class:`.ErrorTrackingBackend`, simulator based on Clifford circuits) and
        ``"stim"`` (:class:`.StimSimulator`, simulator using Stim as
        backend).

        Further devices may be provided as plugins to plaquette. Note that such
        plugins may have to be installed separately to plaquette and that a new
        Python session may have to be started to have such devices be
        recognized after installation.

        Args:
            backend: The name of the backend to use.
            args: Arguments that the backend takes.
            kwargs: Keyword arguments that the backend takes.

        Notes:
            Arguments and keywords arguments meant for the simulator are not
            checked on device creation. Running a circuit will fail if there
            are incorrect arguments passed. See the docs of the backend
            you plan on using for a list of accepted arguments and
            keyword arguments.
        """
        if backend not in recognized_devices:
            raise ValueError(f"Specified backend {backend} is not recognized.")

        self._args, self._kwargs = args, kwargs
        self._backend_name = backend
        self._backend_class = recognized_devices[backend].load()
        self._backend = self._backend_class(*self._args, **self._kwargs)

        self._circuit = None

    @property
    def circuit(self):
        """The underlying quantum circuit to simulate using the backend."""
        return self._backend.circ

    @circuit.setter
    def circuit(self, circuit):
        """Set the underlying quantum circuit to simulate using the backend."""
        self._backend.circ = circuit

    def __iter__(self):
        """Iterate through instructions one-by-one.

        The underlying backend has to define the ``__iter__`` method.
        """
        return self._backend.__iter__()

    def __next__(self):
        """Step to the next gate/instruction in the circuit sequence.

        The underlying backend has to define the ``__next__`` method.
        """
        return self._backend.__next__()

    @property
    def state(self):
        """The underlying quantum state of the backend, if available.

        The underlying backend has to define the ``state`` property.
        """
        return self._backend.state

    @property
    def n_qubits(self):
        """Number of qubits that underlying backend handles.

        The underlying backend has to define the ``n_qubits`` property.
        """
        return self._backend.n_qubits

    def reset_backend(self, *args, **kwargs):
        """Reset the underlying backend."""
        return self._backend.reset(*args, **kwargs)

    def run(
        self,
        circuit: plaq_circuit.Circuit | plaq_circuit.CircuitBuilder,
        *,
        shots=1,
        **kwargs,
    ):
        """Run the given circuit.

        Args:
            circuit: The circuit (or the builder containing it) to be simulated.

        Keyword Args:
            shots: for remote backends, the number of shots to execute the
                circuit with.
            kwargs: backend-specific keyword arguments. For the Clifford
                simulator, the ``after_reset`` keyword argument may be set. If
                ``False``, the returned measurement and erasures will still contain
                any data from previous runs.  Otherwise, both these results and the
                internal state will be reset.
        """
        return self._backend.run(circuit, shots=shots, **kwargs)

    def get_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the samples **after a circuit run**.

        Notes:
            This method assumes that the run method has already been called.
        """
        return self._backend.get_sample()

    @property
    def is_completed(self) -> Optional[list[bool]]:
        """Returns whether the jobs submitted by the device have been completed.

        Notes:
            This method simply returns None for simulators.
        """
        if self._backend_name in local_simulators:
            return None
        return self._backend.is_completed


class AbstractSimulator(metaclass=abc.ABCMeta):
    """Simulator base class.

    .. automethod:: __init__
    """

    @abc.abstractmethod
    def reset(self):
        """Reset this simulator to its default state.

        The simulator will discard all data that it stored which came from
        circuit runs, if any, and will reset any internal state it has to its
        appropriate "zero" state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run(
        self,
        circuit: plaq_circuit.Circuit | plaq_circuit.CircuitBuilder,
        *,
        shots=1,
        **kwargs,
    ):
        """Run the given circuit.

        Args:
            circuit: The Clifford circuit (or the builder containing it) to be
                simulated.

        Keyword Args:
            shots: for remote backends, the number of shots to execute the
                circuit with.
            kwargs: backend-specific keyword arguments. For the Clifford
                simulator, the ``after_reset`` keyword argument may be set. If
                ``False``, the returned measurement and erasures will still contain
                any data from previous runs.  Otherwise, both these results and the
                internal state will be reset.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the samples **after a circuit run**.

        Returns:
            a tuple whose first item is an array of measurement outcomes and
            whose second item is the erasure information. For the Clifford
            simulator, if you draw a new sample with ``after_reset=False``,
            then the returned value will be a concatenation of results of the
            latest sample, plus all other samples since the last time
            ``after_reset`` was ``True``.

        Notes:
            The **measurements** item in the returned tuple is a one-dimensional array
            which contains the following entries, in sequence:

            * ``n_logical_qubits`` results from logical operator measurements for state
              preparation.
            * ``(n_rounds + 1) * n_stabgens`` results from stabilizer generator
              measurements:

              * ``n_stabgens`` results from initial stabilizer generator measurements
                for state preparation.
              * ``n_rounds * n_stabgens`` results from stabilizer generator
                measurements.
            * ``n_logical_qubits`` results from logical operator measurements for state
              verification.

            Here, ``n_logical_qubits`` is the number of logical qubits, ``n_stabgens``
            is the number of stabilizer generators and ``n_rounds`` is the number of
            rounds of stabilizer measurements, all of which are attributes of
            :class:`plaquette.codes.StabilizerCode`.

            The described sequence of measurements is implemented in the circuit
            generator in :meth:`~.QECCircuitGenerator.get_circuit`.

            The **erasure** array specifies, for each data qubit and each round of
            measurements, whether it was erased or not. The shape of this array is
            ``[n_rounds * n_qubits]``.

            To unpack these arrays, you can use
            :meth:`~plaquette.device.MeasurementSample.from_code_and_raw_results`.

            .. todo::
                We need to come up with a way to remove this additional step.
                In theory, a simulator class should immediately return a
                :class:`.MeasurementSample`, and not require this class method.
        """
        raise NotImplementedError()
