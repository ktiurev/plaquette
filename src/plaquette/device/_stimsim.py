# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Interface to Stim for use as circuit simulator.

Stim is available from https://github.com/quantumlib/Stim (Apache 2.0 license).
"""

import typing as t
from typing import Optional

import numpy as np
import stim  # type: ignore

import plaquette
from plaquette import circuit as plaq_circuit
from plaquette import device


def circuit_to_stim(circ: plaq_circuit.Circuit) -> tuple[stim.Circuit, list[bool]]:
    """Convert Clifford circuit in ``plaquette``'s format to Stim's format.

    Args:
        circ: Circuit in ``plaquette``'s format

    Returns:
        a tuple of two elements.

        ``stim_circuit``:
            Circuit in Stim's format
        ``meas_is_erasure``
           For each measurement result, this list contains an entry which specifies
           whether the result signals an erasure (``True``) or a regular
           measurement outcome (:class`False`).
    """
    res = stim.Circuit()
    # For each measuremen result, the following list contains an entry which specifies
    # whether the result signals an erasure or a regular measurement outcome.
    meas_is_erasure: list[bool] = []
    for name, args in circ.gates:
        match name:
            case "X" | "Y" | "Z" | "H" | "R" | "CX" | "CZ":
                res.append(name, args)
            case "M":
                res.append(name, args)
                meas_is_erasure.extend([False] * len(args))
            case "DEPOLARIZE":
                res.append(name, args, 1.0)
            case "E_PAULI":
                assert len(args) == 4
                res.append("PAULI_CHANNEL_1", args[3], args[:3])
            case "E_PAULI2":
                assert len(args) == 17
                res.append("PAULI_CHANNEL_2", args[15:], args[:15])
            case "E_ERASE":
                assert len(args) == 2
                p, target = args
                res.append("HERALDED_ERASE", target, p)
                meas_is_erasure.append(True)
            case "ERROR":
                p, name2, *args2 = args
                if name2 in ("X", "Y", "Z"):
                    if len(args2) != 1:
                        raise ValueError("ERROR ... XYZ only supported on one qubit")
                    res.append(name2 + "_ERROR", args2, p)
                else:
                    raise ValueError(f"ERROR ... {name!r} not supported yet")
            case "ERROR_ELSE" | "ERROR_CONTINUE":
                raise ValueError(f"{name!r} not supported yet")
    return res, meas_is_erasure


class StimSimulator(device.AbstractSimulator):
    """Circuit simulator using Stim as backend.

    .. automethod:: __init__
    """

    def __init__(
        self,
        *,
        stim_seed: Optional[int] = None,
        batch_size: int = 1024,
    ):
        """Create a new Stim-based circuit simulator.

        Args:
            stim_seed: If omitted, a random seed is generated using
                :attr:`plaquette.rng`.
            batch_size: Number of pre-computed samples.

        Stim is more efficient if we compute multiple samples simultaneously. For this
        reason, :attr:`get_sample()` pre-computes a number of ``batch_size`` samples
        whenever it needs to get new samples from Stim.
        """
        self.batch_size = batch_size
        """Batch size for retrieving samples from Stim (retrieving single
        samples is inefficient)."""

        self.stim_seed: int = (
            stim_seed if stim_seed is not None else plaquette.rng.integers(0, 2**63)
        )
        """The last-used sampler."""

        self.circ: plaq_circuit.Circuit | None = None
        """The circuit."""

        self.stim_circ: stim.Circuit | None = None
        """The circuit, converted to Stim's format."""

        self.stim_sampler: stim.CompiledMeasurementSampler | None = None
        """The last-used sampler."""

        self.batch: np.ndarray | None = None
        """Data from current batch."""

        self.batch_remaining: int = 0
        """Remaining unused entries from current batch."""

        self.all_meas: np.ndarray = np.array([])
        """All measurement results of the current batch."""

        self.meas_is_erasure: np.ndarray = np.array([])
        """Determines which measurement results from the Stim circuit are
        actually erasure indicators."""

    def reset(self, new_seed: Optional[int] = None):
        """Compile a new Stim sampler.

        Args:
            new_seed: if not ``None``, this will be set to :attr:`stim_seed`
                and used to build the new sampler.
        """
        if new_seed is not None:
            self.stim_seed = new_seed

        self.circ = None
        self.stim_circ = None
        self.meas_is_erasure = None
        self.stim_sampler = None
        # Erase current batch (if any)
        self.batch = None
        self.batch_remaining = 0

    def run(
        self,
        circuit: plaq_circuit.Circuit | plaq_circuit.CircuitBuilder,
        *,
        shots=1,
        **kwargs,
    ):  # noqa: D102
        if isinstance(circuit, plaq_circuit.CircuitBuilder):
            circ = circuit.circ
        elif isinstance(circuit, plaq_circuit.Circuit):
            circ = circuit
        else:
            raise TypeError(
                "Only a Circuit or a CircuitBuilder can be used in a simulator"
            )

        if shots != 1:
            raise ValueError("The Stim simulator only allows running with shots=1.")

        # A circuit different to the cached one was passed - we create a Stim
        # sampler before sampling the circuit.
        if self.circ is not circ:
            self.circ = circ

            # Convert the circuit to Stim format.
            self.stim_circ, is_erasure = circuit_to_stim(self.circ)
            self.meas_is_erasure = np.array(is_erasure)
            self.stim_sampler = self.stim_circ.compile_sampler(seed=self.stim_seed)

        if self.batch_remaining == 0 and self.stim_sampler is not None:
            self.batch = self.stim_sampler.sample(shots=self.batch_size)
            self.batch_remaining = self.batch_size
        assert self.batch is not None
        self.all_meas = self.batch[-self.batch_remaining]
        self.batch_remaining -= 1

    def get_sample(  # noqa: D102
        self,
    ) -> tuple[np.ndarray, t.Optional[np.ndarray]]:
        # Split results into actual measurements and erasure indications
        meas = self.all_meas[~self.meas_is_erasure]
        qubits_erased = self.all_meas[self.meas_is_erasure]
        if len(qubits_erased) == 0:
            qubits_erased = None
        return meas, qubits_erased
