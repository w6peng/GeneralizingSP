import torch
import warnings
import numpy as np
def bit2float(b, num_e_bits=8, num_m_bits=23, bias=127.):
    """Turn input tensor into float.

        Args:
            b : binary tensor. The last dimension of this tensor should be the
            the one the binary is at.
            num_e_bits : Number of exponent bits. Default: 8.
            num_m_bits : Number of mantissa bits. Default: 23.
            bias : Exponent bias/ zero offset. Default: 127.
        Returns:
            Tensor: Float tensor. Reduces last dimension.

    """
    expected_last_dim = num_m_bits + num_e_bits + 1
    assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                             "should be {}, not {}.".format(
        expected_last_dim, b.shape[-1])

    # check if we got the right type
    dtype = torch.float32
    if expected_last_dim > 32: dtype = torch.float64
    if expected_last_dim > 64:
        warnings.warn("pytorch can not process floats larger than 64 bits, keep"
                      " this in mind. Your result will be not exact.")

    s = torch.index_select(b, -1, torch.arange(0, 1))
    e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits))
    m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                               1 + num_e_bits + num_m_bits))
    # SIGN BIT
    out = ((-1) ** s).squeeze(-1).type(dtype)
    # EXPONENT BIT
    exponents = -torch.arange(-(num_e_bits - 1.), 1.)
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
    out *= 2 ** e_decimal
    # MANTISSA
    matissa = (torch.Tensor([2.]) ** (
        -torch.arange(1., num_m_bits + 1.))).repeat(
        m.shape[:-1] + (1,))
    out *= 1. + torch.sum(m * matissa, dim=-1)
    return out


def float2bit(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
    """Turn input tensor into binary.

        Args:
            f : float tensor.
            num_e_bits : Number of exponent bits. Default: 8.
            num_m_bits : Number of mantissa bits. Default: 23.
            bias : Exponent bias/ zero offset. Default: 127.
            dtype : This is the actual type of the tensor that is going to be
            returned. Default: torch.float32.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.

    """
    ## SIGN BIT
    s = torch.sign(f)
    f = f * s
    # turn sign into sign-bit
    s = (s * (-1) + 1.) * 0.5
    s[s == 0.5] = 1
    s = s.unsqueeze(-1)

    ## EXPONENT BIT
    e_scientific = torch.floor(torch.log2(f))
    e_decimal = e_scientific + bias
    e = integer2bit(e_decimal, num_bits=num_e_bits)
    e[torch.isnan(e)] = 0

    ## MANTISSA
    m1 = integer2bit(f - f % 1, num_bits=num_e_bits)
    m2 = remainder2bit(f % 1, num_bits=bias)
    m = torch.cat([m1, m2], dim=-1)

    dtype = f.type()
    idx = torch.arange(num_m_bits).unsqueeze(0).type(dtype) \
          + (8. - e_scientific).unsqueeze(-1)
    idx = idx.long()
    idx[idx == -9223372036854775808] = 0
    m = torch.gather(m, dim=-1, index=idx)

    return torch.cat([s, e, m], dim=-1).type(dtype)


def remainder2bit(remainder, num_bits=127):
    """Turn a tensor with remainders (floats < 1) to mantissa bits.

        Args:
            remainder : torch.Tensor, tensor with remainders
            num_bits : Number of bits to specify the precision. Default: 127.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.

    """
    dtype = remainder.type()
    exponent_bits = torch.arange(num_bits).type(dtype)
    exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
    out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
    return torch.floor(2 * out)


def integer2bit(integer, num_bits=8):
    """Turn integer tensor to binary representation.

        Args:
            integer : torch.Tensor, tensor with integers
            num_bits : Number of bits to specify the precision. Default: 8.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.

    """
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2

def int_tensor_to_binary_with_sign(tensor, binary_length=7):
    # Ensure tensor is a NumPy array
    data_type = np.int8

    tensor = np.array(tensor, dtype=data_type)

    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Get the shape of the tensor
    shape = tensor.shape

    # Initialize a 3D array to store the binary representation without the sign bit
    binary_tensor = np.zeros(shape + (binary_length,), dtype=data_type)

    # Use bitwise operations to fill the binary_tensor
    for i in range(binary_length):
        binary_tensor[..., binary_length - 1 - i] = (abs(tensor) & (1 << i)) >> i

    # Initialize a 3D array to store the final binary representation with the sign bit
    binary_tensor_with_sign = np.zeros(shape + (binary_length + 1,), dtype=data_type)

    # Add sign bit
    binary_tensor_with_sign[..., 0] = np.where(tensor < 0, 1, 0)

    # Copy the binary representation without the sign bit to the final array
    binary_tensor_with_sign[..., 1:] = binary_tensor

    return binary_tensor_with_sign

def tensor_to_integers(binary_tensor):
    # Extract the sign bit and 7 bits of binary representation
    sign_bit = binary_tensor[..., 0]
    binary_representation = binary_tensor[..., 1:]

    # Convert binary to decimal
    powers_of_two = 2**torch.arange(binary_representation.size(-1) - 1, -1, -1, dtype=binary_representation.dtype)
    decimal_values = torch.sum(binary_representation * powers_of_two, dim=-1)

    # Apply the sign bit
    decimal_values = torch.where(sign_bit == 1, -decimal_values, decimal_values)

    return decimal_values