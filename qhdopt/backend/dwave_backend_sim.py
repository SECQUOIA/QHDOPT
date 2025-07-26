from typing import Tuple, List

from simuq import QSystem, Qubit
from simuq.dwave import DWaveSimProvider
import numpy as np
import time
from qhdopt.utils.decoding_utils import spin_to_bitstring

from qhdopt.backend.backend import Backend


class DWaveBackendSim(Backend):
    """
    Backend implementation for D-Wave simulated annealing using neal.
    This backend doesn't require an API key and runs simulated annealing locally.
    """
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots=100,
                 embedding_scheme="unary",
                 anneal_schedule=None,
                 penalty_coefficient=0,
                 penalty_ratio=0.75,
                 chain_strength_ratio=1.05,
                 **sampler_kwargs):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict,
                         bivariate_dict)
        if anneal_schedule is None:
            anneal_schedule = [[0, 0], [20, 1]]
        self.anneal_schedule = anneal_schedule
        self.penalty_coefficient = penalty_coefficient
        self.penalty_ratio = penalty_ratio
        self.chain_strength_ratio = chain_strength_ratio
        # Store additional kwargs for SimulatedAnnealingSampler
        self.sampler_kwargs = sampler_kwargs

    def calc_penalty_coefficient_and_chain_strength(self) -> Tuple[float, float]:
        """
        Calculates the penalty coefficient and chain strength using self.penalty_ratio.
        """
        if self.penalty_coefficient != 0:
            chain_strength = np.max([5e-2, self.chain_strength_ratio * self.penalty_coefficient])
            return self.penalty_coefficient, chain_strength
          
        qs = QSystem()
        qubits = [Qubit(qs) for _ in range(len(self.qubits))]
        qs.add_evolution(self.S_x(qubits) + self.H_p(qubits, self.univariate_dict, self.bivariate_dict), 1)
        dwp = DWaveSimProvider(**self.sampler_kwargs)
        h, J = dwp.compile(qs, self.anneal_schedule)
        max_strength = np.max(np.abs(list(h) + list(J.values())))
        penalty_coefficient = (
            self.penalty_ratio * max_strength if self.embedding_scheme == "unary" else 0
        )
        # For simulated annealing, chain_strength is not as critical but we keep it for consistency
        chain_strength_multiplier = np.max([1, self.penalty_ratio])
        chain_strength = np.max([5e-2, chain_strength_multiplier * max_strength])
        return penalty_coefficient, chain_strength

    def compile(self, info, override=None):
        penalty_coefficient, chain_strength = self.calc_penalty_coefficient_and_chain_strength()

        if override is not None:
            penalty_coefficient, chain_strength = override

        self.penalty_coefficient, self.chain_strength = penalty_coefficient, chain_strength
        self.qs.add_evolution(
            self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        self.dwp = DWaveSimProvider(**self.sampler_kwargs)
        start_compile_time = time.time()
        self.dwp.compile(self.qs, self.anneal_schedule, chain_strength)
        end_compile_time = time.time()
        info["compile_time"] = end_compile_time - start_compile_time

    def exec(self, verbose: int, info: dict, compile_only=False, override=None) -> List[List[int]]:
        """
        Execute the D-Wave simulated annealing backend using the problem description specified in
        self.univariate_dict and self.bivariate_dict. It uses self.H_p to generate
        the problem hamiltonian and then uses SimuQ's DWaveSimProvider to run simulated annealing.

        Args:
            verbose: Verbosity level.
            info: Dictionary to store information about the execution.
            compile_only: If True, the function only compiles the problem and does not run it.

        Returns:
            raw_samples: A list of raw samples from the simulated annealing.
        """
        self.compile(info, override)

        if verbose > 1:
            self.print_compilation_info()

        if verbose > 1:
            print("Submit Task to Simulated Annealing:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        start_run_time = time.time()
        self.dwave_response = self.dwp.run(shots=self.shots)
        info["backend_time"] = time.time() - start_run_time
        # Simulated annealing runs locally, so machine time equals backend time
        info["average_qpu_time"] = info["backend_time"] / self.shots
        info["time_on_machine"] = info["backend_time"]
        info["overhead_time"] = 0.0

        if verbose > 1:
            print("Received Task from Simulated Annealing:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        if verbose > 0:
            print(f"Backend Simulation Time: {info['time_on_machine']}")
            print(f"Overhead Time: {info['overhead_time']}\n")

        raw_samples = [spin_to_bitstring(result) for result in self.dwp.results()]

        return raw_samples

    def calc_h_and_J(self) -> Tuple[List, dict]:
        """
        Function for debugging to provide h and J which uniquely specify the problem hamiltonian

        Returns:
            h: List of h values
            J: Dictionary of J values
        """
        (
            penalty_coefficient,
            chain_strength,
        ) = self.calc_penalty_coefficient_and_chain_strength()
        self.qs.add_evolution(
            self.S_x(self.qubits) + self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        dwp = DWaveSimProvider(**self.sampler_kwargs)
        return dwp.compile(self.qs, self.anneal_schedule, chain_strength)

    def print_compilation_info(self):
        print("* Compilation information")
        print("Final Hamiltonian:")
        print("(Feature under development; only the Hamiltonian is meaningful here)")
        print(self.qs)
        print(f"Annealing schedule parameter: {self.anneal_schedule}")
        print(f"Penalty coefficient: {self.penalty_coefficient}")
        print(f"Chain strength: {self.chain_strength}")
        print(f"Number of shots: {self.shots}")
        print("Using simulated annealing (no API key required)")
