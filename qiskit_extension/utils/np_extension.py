from functools import reduce

import numpy as np


def tensor_product(*matrices: np.ndarray) -> np.ndarray:
    """Tensor product of a list of square matrix or vector."""
    return reduce(np.kron, matrices)


def inverse_tensor(array: np.ndarray) -> np.ndarray:
    """Inverse tensor product of a square matrix or vector.
    Only accept square matrix or vector with length of 2^n.

    Example:
        input  = A ⊗ B ⊗ C ,
        output = C ⊗ B ⊗ A ,
        (A, B, C are square matrix or vector with length of 2)

    Args:
        array (ndarray): A square matrix or vector.

    Raises:
        ValueError: If the input is not a square matrix or vector.
        ValueError: If the length of input matrix/vector is not 2^n.

    Returns:
        ndarray: The inverse tensor product of input square matrix/vector.
    """
    # If input is a 1d array, extend it to a 2d array
    extend_flag = False
    if len(array.shape) == 1:
        extend_flag = True
        array = array[np.newaxis]

    # Check input validity
    dim_i = array.shape[0]
    dim_j = array.shape[1]
    if dim_i != 1 and dim_j != 1 and dim_i != dim_j:
        raise ValueError("Only accept square matrix or vector.")
    if dim_i & (dim_i - 1) or dim_j & (dim_j - 1):
        raise ValueError("Only accept matrix/vector with length of 2^n.")

    # Linear transform A⊗B⊗...⊗N -> N⊗...⊗B⊗A
    reorder_array = np.zeros_like(array)
    bit_i = int(np.log2(dim_i))
    bit_j = int(np.log2(dim_j))
    for i in range(dim_i):
        # Inverse index (binary), e.g. 1010 -> 0101; 0100 -> 0010
        str_i = bin(i)[2:]
        str_i = "0" * (bit_i - len(str_i)) + str_i
        inverse_i = int(str_i[::-1], 2)
        for j in range(dim_j):
            str_j = bin(j)[2:]
            str_j = "0" * (bit_j - len(str_j)) + str_j
            inverse_j = int(str_j[::-1], 2)
            # Use inverse index to reorder the array
            reorder_array[i, j] = array[inverse_i, inverse_j]

    # If input is a 1d array, squeeze output back to a 1d array
    if extend_flag:
        return reorder_array.squeeze()
    return reorder_array


if __name__ == "__main__":
    array = np.array([[0], [1], [2], [3]])
    print(inverse_tensor(array), "\n")
    array = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    print(inverse_tensor(array), "\n")
    array = np.array([0, 1, 2, 3])
    print(inverse_tensor(array), "\n")
    array = np.array([[0, 1, 2, 3]])
    print(inverse_tensor(array), "\n")
