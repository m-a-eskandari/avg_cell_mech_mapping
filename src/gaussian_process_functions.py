import numpy as np
import LinOT
import MultiScaleOT
import scipy.sparse
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

def compute_linot_tangent(sample_density, reference_position, reference_mass):
    """Initializes MultiScaleOT Sinkhorn solver and computes LogW2 displacement tangent vectors."""

    hierarchy_depth = 7
    number_of_layers = hierarchy_depth + 1

    sample_mass, sample_position = sample_density[0]

    # Initialize multiscale hierarchies for reference and sample
    multi_scale_center = MultiScaleOT.TMultiScaleSetup(reference_position, reference_mass, hierarchy_depth, 0)
    multi_scale_sample = MultiScaleOT.TMultiScaleSetup(sample_position, sample_mass, hierarchy_depth, 0)

    # Cost function and epsilon scaling handler
    cost_function = MultiScaleOT.THierarchicalCostFunctionProvider_SquaredEuclidean(multi_scale_center, multi_scale_sample)
    eps_scaling_handler = MultiScaleOT.TEpsScalingHandler()
    eps_scaling_handler.setupGeometricMultiLayerB(number_of_layers, 1.0, 4.0, 2, 2)

    # Initialize and execute Sinkhorn solver
    sinkhorn_solver = MultiScaleOT.TSinkhornSolverStandard(
        eps_scaling_handler, 0, hierarchy_depth, 1e-3, multi_scale_center, multi_scale_sample, cost_function
    )
    sinkhorn_solver.initialize()
    sinkhorn_solver.solve()

    # Construct CSR kernel sparse matrix and compute LogW2 displacement vectors
    kernel_csr_tuple = sinkhorn_solver.getKernelCSRDataTuple()
    csr_kernel_matrix = scipy.sparse.csr_array(kernel_csr_tuple, shape=(len(reference_mass), len(sample_mass)))

    return LinOT.LogW2(csr_kernel_matrix, reference_position, sample_position)

def matern32(dist, ell=300.0):
    return (1.0 + dist / ell) * np.exp(-dist / ell)

def transform_r_vec(r_vec, r_t_vec, r_t_transformed_vec, ell=300.0, zeta=0.01):
    """Vectorized Matern transformation mapping sample coordinates to average template space."""
    dist_rt_rt = cdist(r_t_vec, r_t_vec)
    C_rt_rt = matern32(dist_rt_rt, ell) + zeta * np.eye(len(r_t_vec))
    cho = cho_factor(C_rt_rt)
    coeff = cho_solve(cho, r_t_transformed_vec)
    dist_r_rt = cdist(r_vec, r_t_vec)
    return matern32(dist_r_rt, ell) @ coeff