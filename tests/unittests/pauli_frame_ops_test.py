# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest as pt

import plaquette
from plaquette.pauli_frame import (
    cx,
    cz,
    depolarize,
    erase,
    hadamard,
    init_pauli_frame,
    maybe_apply_z,
    measure,
    pauli_error_one_qubit,
    pauli_error_two_qubits,
    reset_qubits,
    x,
    y,
    z,
)


class TestBasicFrameOps:
    """Unit testing the Pauli frame ops functions."""

    @pt.mark.parametrize(
        "input_frame, expected_frame, target",
        [
            (np.array([0, 1], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), 0),
            (
                np.array([0, 1, 1, 1], dtype=np.uint8),
                np.array([0, 1, 0, 1], dtype=np.uint8),
                0,
            ),
            (
                np.array([0, 1, 1, 1], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                1,
            ),
            (
                np.array([0, 1, 1, 1, 0, 1], dtype=np.uint8),
                np.array([0, 1, 1, 1, 0, 0], dtype=np.uint8),
                2,
            ),
        ],
    )
    def test_maybe_apply_z(self, input_frame, expected_frame, target, monkeypatch):
        """Test maybe_apply_z."""

        class RNGMock:
            @staticmethod
            def integers(a):
                return 1

        with monkeypatch.context() as m:
            m.setattr(plaquette, "rng", RNGMock)
            res_pauli_frame = maybe_apply_z(input_frame, target)

            assert len(res_pauli_frame) == len(expected_frame)
            assert np.allclose(res_pauli_frame, expected_frame)

    @pt.mark.parametrize(
        "num_qubits, expected",
        [
            (1, np.array([0, 0], dtype=np.uint8)),
            (2, np.array([0, 0, 0, 0], dtype=np.uint8)),
            (3, np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)),
        ],
    )
    def test_init_pauli_frame(self, num_qubits, expected, monkeypatch):
        """Test the initialization of a Pauli frame."""
        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )
            res_frame = init_pauli_frame(num_qubits)

            assert len(res_frame) == len(expected)
            assert np.allclose(res_frame, expected)


class TestQuantumOpsAppliedToFrame:
    @pt.mark.parametrize(
        "input_frame, expected_frame, targets",
        [
            (np.array([0, 0], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), [0]),
            (np.array([1, 0], dtype=np.uint8), np.array([0, 1], dtype=np.uint8), [0]),
            (np.array([0, 1], dtype=np.uint8), np.array([1, 0], dtype=np.uint8), [0]),
            (np.array([1, 1], dtype=np.uint8), np.array([1, 1], dtype=np.uint8), [0]),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 1, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 1], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([1, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 1, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.array([0, 0, 1, 1], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 0, 0, 1], dtype=np.uint8),
                np.array([0, 1, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 0, 1, 0], dtype=np.uint8),
                np.array([1, 0, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 0, 1, 1], dtype=np.uint8),
                np.array([1, 1, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 0, 1, 1], dtype=np.uint8),
                np.array([1, 0, 0, 1], dtype=np.uint8),
                [0],
            ),
            (
                np.array([0, 0, 1, 1], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                [1],
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                [0],
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 1], dtype=np.uint8),
                [1],
            ),
        ],
    )
    def test_hadamard(
        self, input_frame, stable_rgen, expected_frame, targets, monkeypatch
    ):
        """Run some simple circuits and compare with the expected outputs."""
        res = hadamard(input_frame, targets)
        assert np.allclose(res, expected_frame)

    @pt.mark.parametrize(
        "input_frame, expected_frame, targets",
        [
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0],
            ),
            (
                np.array([1, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 1, 1, 0], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                [1, 0],
            ),
        ],
    )
    def test_pauli_gates(
        self, input_frame, expected_frame, stable_rgen, targets, monkeypatch
    ):
        """Run some simple circuits and compare with the expected outputs."""
        res = x(input_frame, targets)
        assert np.allclose(res, expected_frame)

        res = y(input_frame, targets)
        assert np.allclose(res, expected_frame)

        res = z(input_frame, targets)
        assert np.allclose(res, expected_frame)

    @pt.mark.parametrize(
        "input_frame, expected_frame, control_qubits, target_qubits",
        [
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([0, 1, 0, 0], dtype=np.uint8),
                np.array([0, 1, 0, 0], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([1, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 0, 0], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 0], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [1],
                [0],
            ),
            (
                np.array([0, 1, 0, 0], dtype=np.uint8),
                np.array([1, 1, 0, 0], dtype=np.uint8),
                [1],
                [0],
            ),
            (
                np.array([1, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 0], dtype=np.uint8),
                [1],
                [0],
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.array([0, 1, 0, 0], dtype=np.uint8),
                [1],
                [0],
            ),
        ],
    )
    def test_cx(
        self,
        input_frame,
        stable_rgen,
        expected_frame,
        control_qubits,
        target_qubits,
        monkeypatch,
    ):
        """Test the CX operation on the frame."""
        plaquette.rng = stable_rgen
        res = cx(input_frame, control_qubits, target_qubits)
        assert np.allclose(res, expected_frame)

    @pt.mark.parametrize(
        "input_frame, expected_frame, control_qubits, target_qubits",
        [
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([0, 1, 0, 0], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([1, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 1], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.array([1, 1, 1, 1], dtype=np.uint8),
                [0],
                [1],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [1],
                [0],
            ),
            (
                np.array([0, 1, 0, 0], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                [1],
                [0],
            ),
            (
                np.array([1, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 1], dtype=np.uint8),
                [1],
                [0],
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.array([1, 1, 1, 1], dtype=np.uint8),
                [1],
                [0],
            ),
        ],
    )
    def test_cz(
        self,
        input_frame,
        stable_rgen,
        expected_frame,
        control_qubits,
        target_qubits,
        monkeypatch,
    ):
        """Test the CZ operation on the frame."""
        plaquette.rng = stable_rgen
        res = cz(input_frame, control_qubits, target_qubits)
        assert np.allclose(res, expected_frame)

    @pt.mark.parametrize(
        "controlled_op_method, control_qubits, target_qubits",
        [
            (cx, [0], []),
            (cx, [], [1]),
            (cx, [0, 1], [2]),
            (cx, [0], [1, 2]),
            (cz, [0], []),
            (cz, [], [1]),
            (cz, [0, 1], [2]),
            (cz, [0], [1, 2]),
        ],
    )
    def test_control_target_mismatch_cx_cz(
        self,
        controlled_op_method,
        control_qubits,
        target_qubits,
        stable_rgen,
    ):
        """Test that an error is raised if the number of control qubits doesn't
        match the number of target qubits."""
        msg = (
            "The number of control qubits "
            "should be equal to the number of target qubits."
        )
        mock_frame = np.array([0, 0])
        with pt.raises(ValueError, match=msg):
            controlled_op_method(mock_frame, control_qubits, target_qubits)

    @pt.mark.parametrize(
        "input_frame, expected_frame, targets",
        [
            (np.array([0, 0], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), [0]),
            (np.array([1, 0], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), [0]),
            (np.array([0, 1], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), [0]),
            (np.array([1, 1], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), [0]),
            (
                np.array([1, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0],
            ),
            (
                np.array([1, 1, 0, 1], dtype=np.uint8),
                np.array([1, 0, 0, 0], dtype=np.uint8),
                [1],
            ),
            (
                np.array([1, 1, 1, 1], dtype=np.uint8),
                np.array([1, 0, 1, 0], dtype=np.uint8),
                [1],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([1, 0, 1, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([1, 1, 1, 1], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0, 1],
            ),
        ],
    )
    def test_reset_qubits(self, input_frame, expected_frame, targets, monkeypatch):
        """Run some simple circuits and compare with the expected outputs."""
        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )

            res = reset_qubits(input_frame, targets)
            assert np.allclose(res, expected_frame)

    class RNGMock:
        @staticmethod
        def choice(samples, p):
            return 0

    @pt.mark.parametrize(
        "probs",
        [
            (0.1, 0.3, 0.6),
            (0.2, 0.4, 0.3),
            (0.1, 0.2, 0.2),
        ],
    )
    def test_pauli_error_one_q_check_probs(self, probs, monkeypatch, mocker):
        """Check that the correct probs are being passed on correctly."""
        mock_frame = np.array([0, 0])
        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )

            spy = mocker.spy(self.RNGMock, "choice")
            m.setattr(plaquette, "rng", self.RNGMock)

            pauli_error_one_qubit(mock_frame, *probs, 0)
            spy.assert_called_once_with(range(4), p=(1 - sum(probs), *probs))

    @pt.mark.parametrize(
        "input_frame, expected_frame, target, pauli_int",
        [
            (np.array([0, 0], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), 0, 0),
            (np.array([1, 0], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), 0, 1),
            (np.array([1, 1], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), 0, 2),
            (np.array([0, 1], dtype=np.uint8), np.array([0, 0], dtype=np.uint8), 0, 3),
        ],
    )
    def test_pauli_error_one_q_check_output(
        self, input_frame, expected_frame, target, pauli_int, monkeypatch, mocker
    ):
        """Check that the correct output is produced."""

        class RNGMock:
            @staticmethod
            def choice(samples, p):
                return pauli_int

        # These probs don't matter in this test
        arbitrary_probs = (0.1, 0.3, 0.6)
        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )
            m.setattr(plaquette, "rng", RNGMock)

            res = pauli_error_one_qubit(input_frame, *arbitrary_probs, target)
            assert len(res) == len(expected_frame)
            assert np.allclose(res, expected_frame)

    @pt.mark.parametrize(
        "probs",
        [
            [
                0.02,
                0.03,
                0.01,
                0.02,
                0.03,
                0.01,
                0.02,
                0.03,
                0.01,
                0.01,
                0.02,
                0.03,
                0.01,
                0.02,
                0.03,
            ],
            [
                0.02,
                0.03,
                0.01,
                0.02,
                0.03,
                0.01,
                0.02,
                0.03,
                0.01,
                0.01,
                0.02,
                0.03,
                0.01,
                0.02,
                0.03,
            ],
        ],
    )
    def test_pauli_error_two_q_check_probs(self, probs, monkeypatch, mocker):
        """Check that the correct probs are being passed on correctly."""
        mock_frame = np.array([0, 0])
        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )

            spy = mocker.spy(self.RNGMock, "choice")
            m.setattr(plaquette, "rng", self.RNGMock)

            pauli_error_two_qubits(mock_frame, probs + [0, 1])
            spy.assert_called_once_with(range(16), p=(1 - sum(probs), *probs))

    @pt.mark.parametrize(
        "input_frame, expected_frame, target, pauli_int",
        [
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                [0, 1],
                0,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 1, 0, 0], dtype=np.uint8),
                [0, 1],
                1,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 1, 0, 1], dtype=np.uint8),
                [0, 1],
                2,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 1], dtype=np.uint8),
                [0, 1],
                3,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 0], dtype=np.uint8),
                [0, 1],
                4,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 0, 0], dtype=np.uint8),
                [0, 1],
                5,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 0, 1], dtype=np.uint8),
                [0, 1],
                6,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 0, 1], dtype=np.uint8),
                [0, 1],
                7,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 1, 0], dtype=np.uint8),
                [0, 1],
                8,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 1, 0], dtype=np.uint8),
                [0, 1],
                9,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 1, 1], dtype=np.uint8),
                [0, 1],
                10,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 1, 1], dtype=np.uint8),
                [0, 1],
                11,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 1, 0], dtype=np.uint8),
                [0, 1],
                12,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                [0, 1],
                13,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 1, 1, 1], dtype=np.uint8),
                [0, 1],
                14,
            ],
            [
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 1, 1], dtype=np.uint8),
                [0, 1],
                15,
            ],
        ],
    )
    def test_pauli_error_two_q_check_output(
        self, input_frame, expected_frame, target, pauli_int, monkeypatch, mocker
    ):
        """Check that the correct output is produced."""

        class RNGMock:
            @staticmethod
            def choice(samples, p):
                return pauli_int

        # These probs don't matter in this test
        arbitrary_probs = [0.1] * 15
        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )
            m.setattr(plaquette, "rng", RNGMock)

            res = pauli_error_two_qubits(input_frame, arbitrary_probs + target)
            assert len(res) == len(expected_frame)
            assert np.allclose(res, expected_frame)

    @pt.mark.parametrize(
        "input_frame, expected_frame, targets, pauli_int",
        [
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 0, 0], dtype=np.uint8),
                [0, 1],
                [0, 0],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 1, 1], dtype=np.uint8),
                [0, 1],
                [1, 1],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 0, 1, 1], dtype=np.uint8),
                [0, 1],
                [1, 2],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1, 1, 1, 0], dtype=np.uint8),
                [0, 1],
                [1, 0],
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8),
                [0, 1],
                [2, 0],
            ),
        ],
    )
    def test_depolarize_check_output(
        self, input_frame, expected_frame, targets, pauli_int, monkeypatch, mocker
    ):
        """Check that the correct output is produced."""

        class RNGMock:
            def __init__(self, pauli_int):
                self.pauli_int = pauli_int

            def integers(self, m1, m2):
                res = self.pauli_int[0]
                self.pauli_int = self.pauli_int[1:]
                return res

        rng_mock = RNGMock(pauli_int)

        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )
            m.setattr(plaquette, "rng", rng_mock)

            res = depolarize(input_frame, targets)
            assert len(res) == len(expected_frame)
            assert np.allclose(res, expected_frame)

    @pt.mark.parametrize(
        "input_frame, expected_frame, target, pauli_int",
        [
            (
                np.array([0, 0], dtype=np.uint8),
                np.array([0, 0], dtype=np.uint8),
                0,
                0,
            ),
            (
                np.array([0, 0], dtype=np.uint8),
                np.array([1, 0], dtype=np.uint8),
                0,
                1,
            ),
            (
                np.array([0, 0], dtype=np.uint8),
                np.array([1, 1], dtype=np.uint8),
                0,
                2,
            ),
            (
                np.array([0, 0], dtype=np.uint8),
                np.array([0, 1], dtype=np.uint8),
                0,
                3,
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                1,
                0,
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 1, 0, 0], dtype=np.uint8),
                1,
                1,
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 1, 0, 1], dtype=np.uint8),
                1,
                2,
            ),
            (
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([0, 0, 0, 1], dtype=np.uint8),
                1,
                3,
            ),
        ],
    )
    def test_erasure_check_output(
        self, input_frame, expected_frame, target, pauli_int, monkeypatch, mocker
    ):
        """Check that the correct output is produced when an erasure happened."""

        class RNGMock:
            def __init__(self, pauli_int):
                self.pauli_int = pauli_int

            def choice(self, _, p):
                return self.pauli_int

            def random(self):
                return 0

        rng_mock = RNGMock(pauli_int)

        with monkeypatch.context() as m:
            m.setattr(
                plaquette.pauli_frame,
                "maybe_apply_z",
                lambda pauli_frame, target: pauli_frame,
            )
            m.setattr(plaquette, "rng", rng_mock)

            res, qubit_was_erased = erase(input_frame, 1, target)
            assert qubit_was_erased
            assert len(res) == len(expected_frame)
            assert np.allclose(res, expected_frame)

    @staticmethod
    def mock_maybe_apply_z(pauli_frame, qubit):
        return pauli_frame

    @pt.mark.parametrize(
        "ref_sample, pauli_frame, expected_samples, targets",
        [
            (
                np.array([1, 1, 1, 1], dtype=np.uint8),
                np.array([0, 0, 0, 0], dtype=np.uint8),
                np.array([1], dtype=np.uint8),
                [0],
            ),
            (
                np.array([1, 1, 0, 1], dtype=np.uint8),
                np.array([0, 1, 1, 1], dtype=np.uint8),
                np.array([1, 0], dtype=np.uint8),
                [0, 1],
            ),
            (
                np.array([0, 1, 1, 1], dtype=np.uint8),
                np.array([1, 1, 0, 1], dtype=np.uint8),
                np.array([0, 1], dtype=np.uint8),
                [1, 0],
            ),
        ],
    )
    def test_measure(
        self, ref_sample, pauli_frame, expected_samples, targets, monkeypatch, mocker
    ):
        """Run some simple circuits and compare with the expected outputs."""

        spy = mocker.spy(self, "mock_maybe_apply_z")

        with monkeypatch.context() as m:
            m.setattr(plaquette.pauli_frame, "maybe_apply_z", self.mock_maybe_apply_z)

            assert spy.call_count == 0
            res_frame, samples = measure(pauli_frame, ref_sample, targets)

            assert len(samples) == len(expected_samples)
            assert np.allclose(samples, expected_samples)
            assert np.allclose(res_frame, pauli_frame)
            assert spy.call_count == len(targets)
