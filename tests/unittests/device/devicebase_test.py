# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit-tests for plaquette's Device."""

import numpy as np
import pytest as pt
from conftest import (
    yaml_circuit_to_pytest_params,
    yaml_parametrized_circuit_to_pytest_params,
)

import plaquette
from plaquette import Device
from plaquette.circuit import Circuit
from plaquette.circuit.generator import generate_qec_circuit
from plaquette.codes import LatticeCode
from plaquette.errors import QubitErrorsDict


class TestDeviceBaseClifford:
    """Group testing of basic Device functionality with CircuitSimulator."""

    @pt.mark.parametrize(
        "input_circuit, exp_result",
        yaml_circuit_to_pytest_params(
            "sample_circuits/sample_circuits.yaml", "small-codes-without-error"
        ),
    )
    def test_run_circuit(self, input_circuit: str, exp_result: str, stable_rgen):
        """Run some simple circuits and compare with the expected outputs."""
        plaquette.rng = stable_rgen
        circ = Circuit.from_str(input_circuit)
        sim = Device("clifford")
        sim.run(circ)
        sim_res, unused_erasure = sim.get_sample()
        assert "".join(map(str, sim_res)) == exp_result

    @pt.mark.parametrize(
        "param_circuit, result",
        yaml_parametrized_circuit_to_pytest_params(
            "sample_circuits/parametrized_circuits.yaml", "error-parametrize-circuits"
        ),
    )
    def test_run_circuit_error_parametrized_tests(
        self, param_circuit: str, result: str, stable_rgen
    ):
        """Run a bunch of circuits with error info and parametric gates.

        The input parameters to this test case is a "packaged" version of a
        very long and convoluted series of ``pytest.mark.parametrize`` options.
        """
        plaquette.rng = stable_rgen
        circ = Circuit.from_str(param_circuit)
        sim = Device("clifford")
        assert sim.is_completed is None

        sim.run(circ)
        sim_res, unused_erasure = sim.get_sample()
        sim_res2 = "".join(map(str, sim_res))
        assert sim_res2 == result
        assert sim.is_completed is None

    def test_reset(self):
        """Test resetting."""
        dev = Device("clifford")
        code = LatticeCode.make_planar(n_rounds=1, size=4)

        qed: QubitErrorsDict = {
            "pauli": {
                q: {"x": 0.05, "y": 1e-15, "z": 1e-15}
                for q in range(len(code.lattice.dataqubits))
            },
            "erasure": {q: {"p": 0.01} for q in range(len(code.lattice.dataqubits))},
        }
        logical_operator = "Z"
        circuit = generate_qec_circuit(code, qed, {}, logical_operator)

        dev.run(circuit)
        sim_res, unused_erasure = dev.get_sample()

        assert dev._backend.meas_results is not None
        assert dev._backend.erasure != []

        dev.reset_backend()

        assert dev._backend.meas_results is None
        assert dev._backend.erasure == []


class TestDeviceBaseStim:
    """Group testing of basic Device functionality with StimSimulator."""

    @pt.mark.parametrize(
        "kwargs",
        [
            {"stim_seed": 62934814123},
            {"stim_seed": 62934814123, "batch_size": 1},
            {"batch_size": 1},
        ],
    )
    def test_run_circuit(self, kwargs):
        """Run some simple circuits and compare with the expected outputs."""
        plaquette.rng = np.random.default_rng(seed=62934814123)
        circ = Circuit.from_str("CX 0 1 0 2\nM 0 1 2")
        sim = Device("stim", **kwargs)
        assert sim.is_completed is None

        sim.run(circ)
        sim_res, unused_erasure = sim.get_sample()
        assert "".join(map(str, sim_res)) == "FalseFalseFalse"
        assert sim.is_completed is None

    @pt.mark.parametrize(
        "original_seed, new_seed",
        [(None, 1234), (1234, None)],
    )
    def test_reset(self, original_seed, new_seed):
        """Test resetting."""
        dev = Device("stim", stim_seed=original_seed)

        circ = Circuit.from_str("CX 0 1 0 2\nM 0 1 2")
        dev.run(circ)

        if original_seed is not None:
            assert dev._backend.stim_seed is original_seed

        assert dev._backend.batch is not None
        assert dev._backend.batch_remaining != 0

        dev.reset_backend(new_seed)
        if new_seed is not None:
            assert dev._backend.stim_seed is new_seed

        assert dev._backend.batch is None
        assert dev._backend.batch_remaining == 0


class TestDeviceAux:
    """Group testing of auxiliary functionality of the Device class."""

    def test_invalid_backend(self):
        """Test if invalid backend name is inputted."""
        with pt.raises(ValueError, match="is not recognized."):
            Device("some_unrecognized_backend")
