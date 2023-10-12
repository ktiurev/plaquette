# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit-tests for plaquette's error tracker backend."""

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
from plaquette.device._errortrackingbackend import ErrorTrackingBackend
from plaquette.errors import QubitErrorsDict


class TestDeviceErrorTracker:
    """Unit testing the ErrorTrackingBackend class."""

    def test_reset(self):
        """Test resetting."""
        dev = Device("clifford")
        code = LatticeCode.make_planar(n_rounds=1, size=4)

        qed: QubitErrorsDict = {
            "pauli": {
                q: {"x": 0.05, "y": 1e-15, "z": 1e-15}
                for q in range(len(code.lattice.dataqubits))
            },
        }
        logical_operator = "Z"
        circuit = generate_qec_circuit(code, qed, {}, logical_operator)

        dev.run(circuit)
        sim_res, unused_erasure = dev.get_sample()

        assert dev._backend.aux_simulator is not None
        assert dev._backend.pauli_frame is not None
        assert dev._backend.circuit is not None
        assert dev._backend.num_qubits == circuit.number_of_qubits

        dev.reset_backend()

        assert dev._backend.pauli_frame is None
        assert dev._backend.circuit is None
        assert dev._backend.num_qubits == 0
        assert dev._backend.ref_sample is None
        assert dev._backend.aux_simulator is None

        assert not dev._backend.in_error
        assert not dev._backend.any_error_applied
        assert not dev._backend.apply_branch

    def test_run(self, monkeypatch, mocker):
        """Test the run method."""

        # Create a mock CircuitSimulator object
        class CircuitSimulatorMock:
            def __init__(self):
                pass

            def run(self, circuit):
                pass

            def get_sample(self):
                return "mock_ref_sample", "mock_ref_erasure"

        sim = ErrorTrackingBackend()
        mock_aux_sim = CircuitSimulatorMock()

        spy1 = mocker.spy(plaquette.pauli_frame, "init_pauli_frame")

        spy2 = mocker.spy(mock_aux_sim, "run")
        spy3 = mocker.spy(mock_aux_sim, "get_sample")

        # Create a mock Circuit object
        class CircuitMock(plaquette.circuit.Circuit):
            def __init__(self):
                pass

            def without_errors(self):
                return self

            @property
            def gates(self):
                return []

            @property
            def number_of_qubits(self):
                return 2

        circuit_mock = CircuitMock()

        # Monkeypatch the 'aux_simulator' attribute to the mock
        # CircuitSimulator object
        monkeypatch.setattr(
            plaquette.device._errortrackingbackend,
            "CircuitSimulator",
            lambda *args: mock_aux_sim,
        )

        # Call the 'run' method with the mock circuit
        sim.run(circuit_mock)

        assert sim.num_qubits == 2

        # Assert that the methods and attributes are called as expected
        spy1.assert_called_once_with(circuit_mock.number_of_qubits)
        spy2.assert_called_once()
        spy3.assert_called_once()

    def test_create_reference_sample(self, monkeypatch):
        """Test pauli frame simulation reference sample creation."""
        mock_samples = [0, 1, 0]

        class MockCircuitSimulator:
            def run(self, circuit):
                pass

            def get_sample(self):
                return mock_samples

        class MockCircuit:
            def without_errors(self):
                return self

        mock_circuit = MockCircuit()

        # Monkeypatch the methods with the mock implementations
        monkeypatch.setattr(
            plaquette.device._errortrackingbackend,
            "CircuitSimulator",
            MockCircuitSimulator,
        )

        error_tracking_backend = ErrorTrackingBackend()
        samples = error_tracking_backend.create_reference_sample(mock_circuit)

        assert np.allclose(samples, mock_samples)

    def test_perform_pauli_frame_simulation(self, monkeypatch):
        """Test pauli frame simulation method with no erasures and one measurement."""
        mock_gates, mock_samples = [("M", [0, 1, 2])], [0, 1, 0]

        def mock_init_pauli_frame(num_qubits):
            return np.zeros(num_qubits)  # Mock implementation returning zeros

        def mock_measure(frame, ref_sample, args, meas_index):
            return frame, mock_samples  # Mock measurement outcome

        class MockCircuit:
            @property
            def gates(self):
                return mock_gates

        mock_circuit = MockCircuit()

        error_tracking_backend = ErrorTrackingBackend()

        # Monkeypatch the methods with the mock implementations
        monkeypatch.setattr(
            plaquette.pauli_frame, "init_pauli_frame", mock_init_pauli_frame
        )
        monkeypatch.setattr(plaquette.pauli_frame, "measure", mock_measure)

        error_tracking_backend.ref_sample = np.array(mock_samples)

        meas_results, erasure = error_tracking_backend.perform_pauli_frame_simulation(
            mock_circuit
        )

        # Perform assertions to check the behavior of the method
        assert isinstance(meas_results, list)
        assert meas_results == mock_samples
        assert erasure == []

    def test_perform_pauli_frame_simulation_with_erasure(self, monkeypatch):
        """Test pauli frame simulation method with one erasure and one measurement."""
        mock_gates, mock_samples, mock_erasures = (
            [("E_ERASE", (1, 0)), ("M", [0, 1, 2])],
            [0, 1, 0],
            [True],
        )

        def mock_init_pauli_frame(num_qubits):
            return np.zeros(num_qubits)  # Mock implementation returning zeros

        def mock_measure(frame, ref_sample, args, meas_index):
            return frame, mock_samples  # Mock measurement outcome

        def mock_erase(frame, p, target):
            return frame, mock_erasures[0]  # Mock erasure outcome

        monkeypatch.setattr(
            plaquette.pauli_frame, "init_pauli_frame", mock_init_pauli_frame
        )

        class MockCircuit:
            @property
            def gates(self):
                return mock_gates

        mock_circuit = MockCircuit()
        error_tracking_backend = ErrorTrackingBackend()

        error_tracking_backend.ref_sample = np.array(mock_samples)

        # Monkeypatch the methods with the mock implementations
        monkeypatch.setattr(
            plaquette.pauli_frame, "init_pauli_frame", mock_init_pauli_frame
        )
        monkeypatch.setattr(plaquette.pauli_frame, "measure", mock_measure)
        monkeypatch.setattr(plaquette.pauli_frame, "erase", mock_erase)

        meas_results, erasure = error_tracking_backend.perform_pauli_frame_simulation(
            mock_circuit
        )

        # Perform assertions to check the behavior of the method
        assert isinstance(meas_results, list)
        assert meas_results == mock_samples

        assert isinstance(erasure, list)
        assert erasure == mock_erasures


class TestDeviceErrorTrackerCircuits:
    """Group testing of circuits run by ErrorTrackingBackend."""

    @pt.mark.parametrize(
        "input_circuit, exp_result",
        yaml_circuit_to_pytest_params(
            "sample_circuits/sample_circuits.yaml", "small-codes-without-error"
        ),
    )
    def test_run_circuit(
        self,
        input_circuit: str,
        exp_result: str,
        stable_rgen,
        monkeypatch,
    ):
        """Run some simple circuits and compare with the expected outputs."""
        plaquette.rng = stable_rgen

        circ = Circuit.from_str(input_circuit)
        sim = Device("clifford")

        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "init_pauli_frame",
                lambda num_qubits: np.zeros(2 * num_qubits, dtype=np.int32),
            )
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
