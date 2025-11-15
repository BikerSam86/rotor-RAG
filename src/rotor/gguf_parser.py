"""
GGUF format parser for BitNet models.

GGUF is the format used by llama.cpp and bitnet.cpp to store quantized models.
This parser extracts ternary weights from BitNet GGUF files.
"""

import struct
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class GGUFValueType(IntEnum):
    """GGUF metadata value types"""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFTensorType(IntEnum):
    """GGUF tensor types (quantization formats)"""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    # BitNet specific types
    TQ1_0 = 30  # Ternary quantization (BitNet)
    TQ2_0 = 31  # 2-bit ternary


@dataclass
class GGUFTensor:
    """Represents a tensor in GGUF file"""
    name: str
    dims: List[int]
    type: GGUFTensorType
    offset: int
    data: Optional[np.ndarray] = None


class GGUFReader:
    """
    Read GGUF format files (llama.cpp/bitnet.cpp model format).

    GGUF file structure:
    1. Magic number (4 bytes): "GGUF"
    2. Version (4 bytes): uint32
    3. Tensor count (8 bytes): uint64
    4. Metadata count (8 bytes): uint64
    5. Metadata key-value pairs
    6. Tensor info (name, dims, type, offset)
    7. Alignment padding
    8. Tensor data
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.version = None
        self.tensor_count = 0
        self.metadata_count = 0
        self.metadata = {}
        self.tensors: Dict[str, GGUFTensor] = {}
        self.data_offset = 0

    def read(self):
        """Read and parse GGUF file"""
        with open(self.filepath, 'rb') as f:
            # Read header
            self._read_header(f)

            # Read metadata
            self._read_metadata(f)

            # Read tensor info
            self._read_tensor_info(f)

            # Calculate data offset (after alignment)
            self.data_offset = self._align_offset(f.tell(), 32)

    def _read_header(self, f):
        """Read GGUF header"""
        # Magic number
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a valid GGUF file: magic={magic}")

        # Version
        self.version = struct.unpack('<I', f.read(4))[0]
        print(f"GGUF version: {self.version}")

        # Tensor count
        self.tensor_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Tensor count: {self.tensor_count}")

        # Metadata count
        self.metadata_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Metadata count: {self.metadata_count}")

    def _read_string(self, f) -> str:
        """Read length-prefixed string"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, value_type: GGUFValueType):
        """Read a value of specified type"""
        if value_type == GGUFValueType.UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return struct.unpack('<?', f.read(1))[0]
        elif value_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif value_type == GGUFValueType.ARRAY:
            # Read array type
            array_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
            # Read array length
            array_len = struct.unpack('<Q', f.read(8))[0]
            # Read array elements
            return [self._read_value(f, array_type) for _ in range(array_len)]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_metadata(self, f):
        """Read metadata key-value pairs"""
        for _ in range(self.metadata_count):
            # Read key
            key = self._read_string(f)

            # Read value type
            value_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])

            # Read value
            value = self._read_value(f, value_type)

            self.metadata[key] = value

    def _read_tensor_info(self, f):
        """Read tensor information"""
        for _ in range(self.tensor_count):
            # Tensor name
            name = self._read_string(f)

            # Number of dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]

            # Dimensions
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]

            # Tensor type
            tensor_type = GGUFTensorType(struct.unpack('<I', f.read(4))[0])

            # Offset in data section
            offset = struct.unpack('<Q', f.read(8))[0]

            # Create tensor object
            tensor = GGUFTensor(
                name=name,
                dims=dims,
                type=tensor_type,
                offset=offset
            )

            self.tensors[name] = tensor

    def _align_offset(self, offset: int, alignment: int) -> int:
        """Align offset to specified boundary"""
        return (offset + alignment - 1) // alignment * alignment

    def get_tensor(self, name: str) -> Optional[GGUFTensor]:
        """Get tensor by name"""
        return self.tensors.get(name)

    def load_tensor_data(self, name: str) -> np.ndarray:
        """Load tensor data from file"""
        tensor = self.tensors.get(name)
        if tensor is None:
            raise ValueError(f"Tensor '{name}' not found")

        with open(self.filepath, 'rb') as f:
            # Seek to tensor data
            f.seek(self.data_offset + tensor.offset)

            # Read based on type
            if tensor.type in (GGUFTensorType.TQ1_0, GGUFTensorType.TQ2_0):
                # BitNet ternary format
                # This is packed: 4 weights per byte (2 bits each)
                total_elements = np.prod(tensor.dims)
                n_bytes = (total_elements + 3) // 4
                packed_data = np.frombuffer(f.read(n_bytes), dtype=np.uint8)

                return packed_data.reshape(tensor.dims[0], -1) if len(tensor.dims) == 2 else packed_data

            elif tensor.type == GGUFTensorType.F32:
                # Float32
                total_elements = np.prod(tensor.dims)
                data = np.frombuffer(f.read(total_elements * 4), dtype=np.float32)
                return data.reshape(tensor.dims)

            elif tensor.type == GGUFTensorType.F16:
                # Float16
                total_elements = np.prod(tensor.dims)
                data = np.frombuffer(f.read(total_elements * 2), dtype=np.float16)
                return data.reshape(tensor.dims)

            else:
                raise NotImplementedError(f"Tensor type {tensor.type} not yet implemented")

    def list_tensors(self) -> List[str]:
        """List all tensor names"""
        return list(self.tensors.keys())

    def print_summary(self):
        """Print summary of GGUF file contents"""
        print(f"\nGGUF File Summary:")
        print(f"  Version: {self.version}")
        print(f"  Tensors: {self.tensor_count}")
        print(f"  Metadata entries: {self.metadata_count}")

        print(f"\nKey Metadata:")
        for key, value in list(self.metadata.items())[:10]:
            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")

        print(f"\nTensors (first 20):")
        for i, (name, tensor) in enumerate(list(self.tensors.items())[:20]):
            dims_str = 'x'.join(map(str, tensor.dims))
            print(f"  {i+1}. {name}: {dims_str}, type={tensor.type.name}")

        if self.tensor_count > 20:
            print(f"  ... and {self.tensor_count - 20} more tensors")


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gguf_parser.py <path_to_gguf_file>")
        print("\nExample:")
        print("  python gguf_parser.py C:\\Users\\samho\\Desktop\\BitNet-2B-model\\ggml-model-i2_s.gguf")
        sys.exit(1)

    gguf_path = sys.argv[1]
    print(f"Reading GGUF file: {gguf_path}")

    # Parse file
    reader = GGUFReader(gguf_path)
    reader.read()

    # Print summary
    reader.print_summary()

    # Try to load a tensor
    if reader.tensors:
        first_tensor_name = list(reader.tensors.keys())[0]
        print(f"\nLoading first tensor: {first_tensor_name}")
        try:
            data = reader.load_tensor_data(first_tensor_name)
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Size: {data.nbytes} bytes")
        except Exception as e:
            print(f"  Error loading: {e}")
