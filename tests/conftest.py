# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import yaml


@pytest.fixture(scope="module")
def stable_rgen():
    return np.random.default_rng(seed=123123456)


def load_strings(file_path):
    with open(file_path, "r") as file:
        contents = file.read()
    return contents.split("\n\n")


def yaml_circuit_to_pytest_params(
    yaml_path: str, circuit_suite_key: str
) -> list[tuple]:
    """Parse non-parametrized circuit to pytests params.

    Args:
        yaml_path: str
        circuit_suite_key: str

    Returns:
        params_list : parameter list that is passed onto pytest.mark.parametrize
    """
    circuit_suite: dict[str, dict[str, str]]
    try:
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]
    except FileNotFoundError:
        yaml_path = "tests/unittests/device/" + yaml_path
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]
    params_list: list[tuple] = []
    for params in circuit_suite.values():
        params_list.append(tuple(params.values()))

    return params_list


def yaml_parametrized_circuit_to_pytest_params(
    yaml_path: str, circuit_suite_key: str
) -> list[tuple]:
    """Parse non-parametrized circuit to pytests params.

    Args:
        yaml_path: str
        circuit_suite_key: str

    Returns:
        params_list: parameter list that is passed onto
            ``pytest.mark.parametrize``.
    """
    try:
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]
    except FileNotFoundError:
        yaml_path = "tests/unittests/device/" + yaml_path
        circuit_suite = yaml.load(open(yaml_path), yaml.SafeLoader)[circuit_suite_key]

    assert isinstance(
        circuit_suite, list
    ), "Please make sure, yaml is correctly specified"
    params_list: list[tuple] = []

    for dict_ in circuit_suite:
        assert isinstance(dict_, dict)
        key = list(dict_.keys())[0]  # this is of length 1 only
        circ_template: str = dict_[key]["circuit-template"]
        params: list = dict_[key]["params"]
        expected_output: list = dict_[key]["expected-output"]
        assert len(params) == len(
            expected_output
        ), "Please make sure number of parameters and expected outputs are same"
        for index in range(len(params)):
            params_list.append(
                (circ_template.format(*params[index]), expected_output[index])
            )

    return params_list
