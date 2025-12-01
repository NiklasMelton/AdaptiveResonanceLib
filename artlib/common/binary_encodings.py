import numpy as np


def binarize_features(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data into `n_bits` using uniform quantization.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for binarization.

    Returns:
        np.ndarray: A binarized representation of the input data with shape (n, m * n_bits).

    """
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize to [0, 1]
    normalized_data = (data - min_vals) / ranges

    # Quantize into `2^n_bits` levels
    quantized_data = (normalized_data * (2**n_bits - 1)).astype(int)

    # Convert to binary representation
    binarized_data = np.unpackbits(
        quantized_data[:, :, None].astype(np.uint8),
        axis=2,
        count=n_bits,
        bitorder="little",
    )

    return binarized_data.reshape(n, m * n_bits)


def int_to_gray(n: np.ndarray) -> np.ndarray:
    """Converts an integer array to its corresponding Gray code representation.

    Parameters:
        n (np.ndarray): Array of integer values.

    Returns:
        np.ndarray: Gray-coded integer values.

    """
    return n ^ (n >> 1)  # XOR with right-shifted self to compute Gray code


def binarize_features_gray(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data into `n_bits` using Gray-coded quantization.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for binarization.

    Returns:
        np.ndarray: A binarized Gray-coded representation of the input data with shape (n, m * n_bits).

    """
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize to [0, 1]
    normalized_data = (data - min_vals) / ranges

    # Quantize into `2^n_bits` levels
    quantized_data = (normalized_data * (2**n_bits - 1)).astype(int)

    # Convert to Gray code
    gray_coded_data = int_to_gray(quantized_data)

    # Convert to binary representation
    binarized_data = np.unpackbits(
        gray_coded_data[:, :, None].astype(np.uint8),
        axis=2,
        count=n_bits,
        bitorder="little",
    )

    return binarized_data.reshape(n, m * n_bits)


def binarize_features_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data using thermometer encoding.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for thermometer encoding.

    Returns:
        np.ndarray: A thermometer-coded representation of the input data with shape (n, m * n_bits).

    """
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize to [0, 1]
    normalized_data = (data - min_vals) / ranges

    if n_bits == 1:
        return (normalized_data > 0.5).astype(np.uint8)

    # Quantize into `n_bits` levels (instead of `2^n_bits` levels)
    quantized_data = np.floor(normalized_data * n_bits).astype(int)

    # Generate thermometer encoding: fill from left to right
    thermometer_encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for i in range(n_bits):
        thermometer_encoded[:, :, i] = (quantized_data > i).astype(np.uint8)

    return thermometer_encoded.reshape(n, m * n_bits)


def binarize_features_log_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data using logarithmic thermometer encoding.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for encoding.

    Returns:
        np.ndarray: A binary encoded representation of the input data with shape (n, m * n_bits).

    """
    if n_bits <= 0:
        raise ValueError("n_bits must be a positive integer.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize data to [0, 1]
    normalized_data = (data - min_vals) / ranges

    if n_bits == 1:
        return (normalized_data > 0.5).astype(np.uint8)

    # Map to a logarithmic scale: Log base 2 ensures better space efficiency
    quantized_data = np.floor(
        np.log1p(normalized_data * (np.e - 1)) / np.log(np.e) * (n_bits - 1)
    ).astype(int)

    # Generate logarithmic thermometer encoding
    encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for i in range(n_bits):
        encoded[:, :, i] = (quantized_data >= i).astype(
            np.uint8
        )  # Logarithmic activation

    return encoded.reshape(n, m * n_bits)


def determine_k_active(data: np.ndarray, n_bits: int, min_k=2, max_k=None):
    """Determines the number of active bits dynamically based on data variance.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Total number of bits in the SDR encoding.
        min_k (int): Minimum number of active bits.
        max_k (int): Maximum number of active bits (default: sqrt(n_bits)).

    Returns:
        np.ndarray: Array of k_active values for each feature.

    """
    if max_k is None:
        max_k = int(np.sqrt(n_bits))  # Default max_k is sqrt(n_bits)

    feature_variance = np.var(data, axis=0)  # Compute variance for each feature
    norm_variance = feature_variance / (np.max(feature_variance) + 1e-8)  # Normalize

    # Map variance to active bit range using logarithmic scaling
    k_active_values = np.round(
        min_k + (max_k - min_k) * np.log1p(norm_variance) / np.log1p(1)
    ).astype(int)

    return np.clip(k_active_values, min_k, max_k)  # Ensure within range


def binarize_features_sdr(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data using SDR with dynamically determined active
    bits.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Total number of bits in the SDR encoding.

    Returns:
        np.ndarray: A binary encoded representation of the input data with shape (n, m * n_bits).

    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize data to [0, 1]
    normalized_data = (data - min_vals) / ranges

    # Determine k_active dynamically per feature
    k_active_values = determine_k_active(data, n_bits)
    print("K_ACTIVE", k_active_values.max())
    k_active_values = np.array([5])

    # Scale to integer range [0, n_bits - max(k_active)]
    quantized_data = np.floor(
        normalized_data * (n_bits - k_active_values.max())
    ).astype(int)

    # Generate SDR encoding
    encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for j in range(m):  # Process each feature separately
        for i in range(n):  # Process each sample
            k_active = k_active_values[j]
            start_pos = quantized_data[i, j]  # Compute start position
            active_indices = (
                np.arange(start_pos, start_pos + k_active) % n_bits
            )  # Wrap around
            encoded[i, j, active_indices] = 1  # Activate k bits

    return encoded.reshape(n, m * n_bits)


def binarize_features_adaptive_sdr(
    data: np.ndarray, n_bits: int, min_k=2, max_k=None
) -> np.ndarray:
    """Binarizes each feature in the data using an adaptive Sparse Distributed
    Representation (SDR).

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Total number of bits in the SDR encoding.
        min_k (int): Minimum number of active bits.
        max_k (int): Maximum number of active bits (default: sqrt(n_bits)).

    Returns:
        np.ndarray: A binary encoded representation of the input data with shape (n, m * n_bits).

    """
    if max_k is None:
        max_k = int(np.sqrt(n_bits))  # Default max_k is sqrt(n_bits)

    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize data to [0, 1]
    normalized_data = (data - min_vals) / ranges

    # Determine adaptive k_active based on variance
    feature_variance = np.var(data, axis=0)
    norm_variance = feature_variance / (np.max(feature_variance) + 1e-8)
    k_active_values = np.round(min_k + (max_k - min_k) * norm_variance).astype(int)

    # Generate SDR encoding
    encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for j in range(m):  # Process each feature separately
        for i in range(n):  # Process each sample
            k_active = k_active_values[j]
            possible_positions = np.arange(n_bits)
            np.random.shuffle(possible_positions)  # Randomly distribute bits
            active_indices = np.sort(
                possible_positions[:k_active]
            )  # Select k active bits
            encoded[i, j, active_indices] = 1

    return encoded.reshape(n, m * n_bits)


from scipy.stats import norm


def binarize_features_gaussian_thermometer(data: np.ndarray, n_bits: int) -> np.ndarray:
    """Binarizes each feature in the data using a Gaussian-based thermometer encoding.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for encoding.

    Returns:
        np.ndarray: A binary encoded representation of the input data with shape (n, m * n_bits).

    """
    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize data to [0, 1]
    normalized_data = (data - min_vals) / ranges

    if n_bits == 1:
        return (normalized_data > 0.5).astype(np.uint8)

    # Define Gaussian activation centers
    bit_positions = np.linspace(0, 1, n_bits)

    # Compute Gaussian probability per value
    encoded = np.zeros((n, m, n_bits), dtype=np.uint8)
    sigma = 0.15  # Controls width of the Gaussian spread

    for i in range(n):
        for j in range(m):
            probs = norm.pdf(bit_positions, loc=normalized_data[i, j], scale=sigma)
            probs /= probs.max()  # Normalize to [0, 1]
            encoded[i, j] = (probs > 0.5).astype(np.uint8)  # Apply threshold

    return encoded.reshape(n, m * n_bits)


def binarize_features_tanh_thermometer(
    data: np.ndarray, n_bits: int, a=4.0
) -> np.ndarray:
    """Binarizes each feature in the data using a tanh-based logarithmic thermometer
    encoding.

    Parameters:
        data (np.ndarray): Input array of shape (n, m), where n is the number of samples and m is the number of features.
        n_bits (int): Number of bits to use for encoding.
        a (float): Tanh scaling parameter; higher values make it more logarithmic.

    Returns:
        np.ndarray: A binary encoded representation of the input data with shape (n, m * n_bits).

    """
    n, m = data.shape
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # Avoid division by zero in case of constant features
    ranges = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1)

    # Normalize data to [0, 1]
    normalized_data = (data - min_vals) / ranges

    # Generate thermometer activation thresholds using tanh-based mapping
    bit_positions = np.linspace(0, 1, n_bits)
    transformed_positions = (np.tanh(a * (bit_positions - 0.5)) + 1) / 2

    # Generate encoding
    encoded = np.zeros((n, m, n_bits), dtype=np.uint8)

    for i in range(n):
        for j in range(m):
            encoded[i, j] = (transformed_positions <= normalized_data[i, j]).astype(
                np.uint8
            )

    return encoded.reshape(n, m * n_bits)
