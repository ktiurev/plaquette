.. Copyright 2023, QC Design GmbH and the plaquette contributors
   SPDX-License-Identifier: Apache-2.0


How to create a new backend?
============================

The error correction pipeline of plaquette involves running the quantum circuit that represents a quantum error correcting code. To run the quantum circuit on a quantum device, plaquette provides the :class:`.Device` object which can be used with various backends, for example with the Clifford local backend:

.. code-block:: python

    from plaquette import Device
    device = Device("clifford")
    device.run(circuit)

The :class:`.Device` object maintains a backend object under the hood. While
plaquette comes with built-in backends, it is also extensible and new backends
can be developed and integrated into plaquette.

A new backend can be created via the following steps:

1. Create a new backend class (we suggest naming the backend class such that it has ``Backend`` as a suffix in its name);
2. Implement the set of required (and in addition the optional) backend methods and properties (as detailed below);
3. Package your new backend into a separate Python package;
4. In the ``pyproject.toml`` file of the Python package add a new section:

   .. code-block:: toml

       [project.entry-points."plaquette.device"]
       my_backend_name = "plaquette_my_backend.backend:MyBackend"

   where ``my_backend_name`` will be the name that can be passed to :class:`.Device` to use the new backend, the new backend package name is ``plaquette_my_backend`` and the backend implementation is represented with the ``MyBackend`` class which is placed in the ``backend`` module of the package.

Refer to the `plaquette-ibm-backend
<https://github.com/qc-design/plaquette-ibm-backend>`_ package for further packaging details.

Backend API
-----------

The set of required backend methods and their signatures are:

* The ``run`` method to run a quantum circuit;

  .. code-block:: python

      run(self, circuit: plaq_circuit.Circuit | plaq_circuit.CircuitBuilder, *, shots=1)

* The ``get_sample`` method to return the samples **after a circuit run**.

  .. code-block:: python

      get_sample(self) -> Tuple[List[Union[List[Any], Any]], Optional[List[None]]]

There are additional optional backend methods and properties for certain backends:

* Remote backends may implement the ``is_completed(self) -> List[bool]``
  property to determine which jobs have been completed for the list of
  submitted jobs;
* Simulator backends that allow obtaining the underlying quantum state may
  either define a ``state(self) -> device.QuantumState`` property or maintain a
  ``self.state: device.QuantumState`` attribute to allow :class:`.Device`
  access the quantum state;
* Simulator backends may also either define a ``n_qubits(self) -> int`` property or
  maintain a ``self.n_qubits: int`` attribute to describe the number of qubits
  of the quantum circuit.

Furthermore, when creating a :class:`.Device` object all additional arguments and keyword arguments are used to create the underlying backend object. Therefore, methods and properties implemented in new backends may take custom arguments that can simply be passed when creating a device.

If you have any questions or suggestions related to creating new backends, feel free to open a `new GitHub issue <https://github.com/qc-design/plaquette/issues/new/choose>`_!
