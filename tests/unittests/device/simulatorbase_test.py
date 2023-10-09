# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit-tests for plaquette's CircuitSimulator."""

import numpy as np
import pytest as pt
from conftest import (
    yaml_circuit_to_pytest_params,
    yaml_parametrized_circuit_to_pytest_params,
)

import plaquette
from plaquette import pauli
from plaquette.circuit import Circuit
from plaquette.device._circuitsim import CircuitSimulator


class TestCircuitSimulatorBase:
    """Group testing of basic CircuitSimulator functionality."""

    @pt.mark.skip
    def test__handle_error(self):
        pass

    @pt.mark.parametrize(
        "name, params, error_message",
        [
            ("RX", (1, 2), "Unknown gate 'RX'"),
            ("CRX", (1, 2), "Unknown gate 'CRX'"),
            ("ERROR_CONTINUES", ("X", 1, 2), "Unknown gate 'ERROR_CONTINUES'"),
            ("ERROR_IF", ("RY", 1, 2), "Unknown gate 'ERROR_IF'"),
        ],
    )
    def test__handle_error_failures(
        self, name: str, params: tuple, error_message: str, stable_rgen
    ):
        """Test that the handling of an unknown error gate fails."""
        plaquette.rng = stable_rgen
        circ = Circuit()
        circ.gates.append((name, params))
        c = CircuitSimulator()
        with pt.raises(ValueError) as error:
            c._handle_error(name, params)

        assert str(error.value) == error_message

    # unclear how to write tests as it invokes tableausim
    @pt.mark.skip
    def test__handle_gate(self):
        pass

    @pt.mark.parametrize(
        "name, params, error_message, err_t",
        [
            ("RX", (1, 2), "Unknown gate 'RX' (this should not happen)", ValueError),
            ("CRX", (1, 2), "Unknown gate 'CRX' (this should not happen)", ValueError),
            (
                "ERROR_CONTINUE",
                ("CRX", 1, 2),
                "Unknown gate 'ERROR_CONTINUE' (this should not happen)",
                ValueError,
            ),
            (
                "ERROR",
                ("RY", 1, 2),
                "Unknown gate 'ERROR' (this should not happen)",
                ValueError,
            ),
        ],
    )
    def test__handle_gate_failures(
        self,
        name: str,
        params: tuple,
        error_message: str,
        stable_rgen,
        err_t: Exception,
    ):
        """Test that the handling of an unknown, non-error gate fails."""
        plaquette.rng = stable_rgen
        circ = Circuit()
        circ.gates.append((name, params))
        c = CircuitSimulator()
        c.state = pauli.zero_state(1)
        with pt.raises(err_t) as error:
            c._handle_gate(name, params)
        assert str(error.value) == error_message

    # unlcear how to write tests as it invokes tableausim
    @pt.mark.skip
    def test__run_gate(self):
        """TBA."""
        pass

    @pt.mark.parametrize(
        "name, params, error_message",
        [
            ("RX", (1, 2), "Unknown gate 'RX' (this should not happen)"),
            ("CRX", (1, 2), "Unknown gate 'CRX' (this should not happen)"),
            ("ERROR_CONTINUE", ("CRX", 1, 2), "ERROR_CONTINUE not valid here"),
            ("ERROR_ELSE", ("CRX", 1, 2), "ERROR_ELSE not valid here"),
            ("ERROR", (1.0, "RY", 1, 2), "Unknown gate 'RY' (this should not happen)"),
            # 1.0 is so that self.rng.random() is less the error prob always to see if
            # the right error is being raised
            ("ERROR_CONTINUES", ("X", 1, 2), "Unknown gate 'ERROR_CONTINUES'"),
            ("ERROR_IF", ("RY", 1, 2), "Unknown gate 'ERROR_IF'"),
        ],
    )
    def test__run_gate_failures(
        self,
        name: str,
        params: tuple,
        error_message: str,
        stable_rgen: np.random.Generator,
    ):
        """Make sure we catch weird gate names."""
        plaquette.rng = stable_rgen
        circ = Circuit()
        circ.gates.append((name, params))
        c = CircuitSimulator()
        c.state = pauli.zero_state(1)
        with pt.raises(ValueError) as error:
            c._run_gate(name, params)
        assert str(error.value) == error_message

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
        sim = CircuitSimulator()
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
        sim = CircuitSimulator()
        sim.run(circ)
        sim_res, unused_erasure = sim.get_sample()
        sim_res2 = "".join(map(str, sim_res))
        assert sim_res2 == result
