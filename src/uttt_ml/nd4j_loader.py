"""
Pure-Python reader for the ND4J/DL4J `.bin` DataSet format.

The original Java project used `DataSet.save(file)` / `DataSet.load(file)`
from `org.nd4j.linalg.dataset.DataSet` to persist training data. Those
files are binary and follow a custom layout — there is no public Python
reader — so this module implements a minimal parser that handles the
specific variant produced by the project.

## Outer DataSet wrapper

From `org.nd4j.linalg.dataset.DataSet.save(OutputStream)`:

    dos.writeByte(included)                     # 1-byte bitmask
    if features != null:  Nd4j.write(features, dos)
    if labels   != null:  Nd4j.write(labels,   dos)
    # optional: feature mask, label mask, metadata, label names

Bitmask bits:
    BITMASK_FEATURES_PRESENT           = 0x01
    BITMASK_LABELS_PRESENT             = 0x02
    BITMASK_LABELS_SAME_AS_FEATURES    = 0x04
    BITMASK_FEATURE_MASK_PRESENT       = 0x08
    BITMASK_LABELS_MASK_PRESENT        = 0x10
    BITMASK_METADATA_PRESET            = 0x20
    BITMASK_LABEL_NAME_PRESET          = 0x40

Every `.bin` file produced by the Ultimate Tic Tac Toe project has
bitmask 0x03 — features and labels present, everything else absent.

## Inner Nd4j.write(INDArray) layout

An INDArray in the "MIXED_DATA_TYPES" format (newer ND4J, which is what
DL4J 1.0.0-M2.1 produces) is serialized as three concatenated blocks:

    1. Shape-info block:
         writeUTF("MIXED_DATA_TYPES")   # 2-byte length + UTF-8
         writeLong(shape_info_length)   # number of longs that follow
         writeUTF(shape_dtype_name)     # always "LONG" for this project
         shape_info_length * writeLong  # shape buffer (see libnd4j layout)

       The shape buffer layout is:
         [0]                rank
         [1 .. rank]        shape dimensions
         [rank+1 .. 2*rank] strides
         [2*rank+1]         options (encodes dtype and flags — ignored here)
         [2*rank+2]         element-wise stride
         [2*rank+3]         order ('c' or 'f', as an ASCII value in a long)
       So the shape buffer length is 2*rank + 4.

    2. Data-type tag block:
         writeUTF("MIXED_DATA_TYPES")

    3. Data block:
         writeLong(element_count)       # total number of values
         writeUTF(data_dtype_name)      # always "DOUBLE" for this project
         element_count * 8 raw bytes    # big-endian IEEE 754 doubles

Everything is big-endian (Java `DataOutputStream` default). The parser
returns NumPy arrays in the standard row-major layout, reshaping the
flat data bytes according to the shape read from the shape-info block.

## Why pure Python

We want the training pipeline to work without a JVM dependency. This
parser only has to handle DOUBLE data with LONG shape metadata, which
is the exact variant DL4J 1.0.0-M2.1 writes for this project. If we
ever encounter a variant with different dtypes, the parser will raise
with a clear error message pointing at the offending file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Bitmask constants — mirror the values in DataSet.java
# ----------------------------------------------------------------------

BITMASK_FEATURES_PRESENT = 0x01
BITMASK_LABELS_PRESENT = 0x02
BITMASK_LABELS_SAME_AS_FEATURES = 0x04
BITMASK_FEATURE_MASK_PRESENT = 0x08
BITMASK_LABELS_MASK_PRESENT = 0x10
BITMASK_METADATA_PRESET = 0x20
BITMASK_LABEL_NAME_PRESET = 0x40


# ----------------------------------------------------------------------
# Low-level primitives — Java DataInputStream compatibility
# ----------------------------------------------------------------------

def _read_byte(buf: bytes, offset: int) -> tuple[int, int]:
    """Read one signed byte — mirrors `DataInputStream.readByte()`."""
    return buf[offset], offset + 1


def _read_long(buf: bytes, offset: int) -> tuple[int, int]:
    """Read a big-endian 8-byte signed long — mirrors `DataInputStream.readLong()`."""
    value = struct.unpack_from(">q", buf, offset)[0]
    return value, offset + 8


def _read_utf(buf: bytes, offset: int) -> tuple[str, int]:
    """Read a Java modified-UTF-8 string — mirrors `DataInputStream.readUTF()`.

    For our purposes (ASCII tags like "MIXED_DATA_TYPES", "LONG", "DOUBLE")
    standard UTF-8 decoding is equivalent to Java's modified UTF-8, so
    we just decode with `errors="strict"` and trust the tags.
    """
    length = struct.unpack_from(">H", buf, offset)[0]
    start = offset + 2
    end = start + length
    return buf[start:end].decode("utf-8"), end


# ----------------------------------------------------------------------
# High-level parser
# ----------------------------------------------------------------------

@dataclass
class Nd4jIndArray:
    """A parsed INDArray: its shape and its raw data as a NumPy array."""

    shape: tuple[int, ...]
    data: np.ndarray  # already reshaped to `shape`


def _parse_indarray(buf: bytes, offset: int) -> tuple[Nd4jIndArray, int]:
    """Read a single Nd4j.write(INDArray) block starting at `offset`.

    Returns the parsed array and the byte offset just past its end.
    """
    # Block 1: shape-info
    tag, offset = _read_utf(buf, offset)
    if tag != "MIXED_DATA_TYPES":
        raise ValueError(
            f"Unsupported ND4J serialization format tag: {tag!r} "
            "(only MIXED_DATA_TYPES is implemented)"
        )

    shape_info_len, offset = _read_long(buf, offset)
    shape_dtype, offset = _read_utf(buf, offset)
    if shape_dtype != "LONG":
        raise ValueError(
            f"Unsupported shape-info dtype: {shape_dtype!r} "
            "(only LONG is implemented)"
        )

    shape_info: list[int] = []
    for _ in range(shape_info_len):
        value, offset = _read_long(buf, offset)
        shape_info.append(value)

    # Decode the libnd4j shape buffer layout. We only actually need
    # the rank and the shape dimensions; strides, options, ews, and
    # order are irrelevant for us because the raw data buffer is
    # stored contiguously in C (row-major) order, which is NumPy's
    # default.
    rank = int(shape_info[0])
    shape = tuple(int(d) for d in shape_info[1 : 1 + rank])
    # shape_info also contains strides, options, ews, and order
    # (shape_info[1+rank : 2*rank+4]) — we validate length but ignore
    # the values.
    expected = 2 * rank + 4
    if len(shape_info) != expected:
        raise ValueError(
            f"Malformed shape info buffer: expected {expected} longs "
            f"for rank {rank}, got {len(shape_info)}"
        )

    # Block 2: inner tag. We've seen only "MIXED_DATA_TYPES" here.
    inner_tag, offset = _read_utf(buf, offset)
    if inner_tag != "MIXED_DATA_TYPES":
        raise ValueError(
            f"Unsupported ND4J data block tag: {inner_tag!r}"
        )

    # Block 3: data count + dtype + raw bytes
    data_count, offset = _read_long(buf, offset)
    data_dtype, offset = _read_utf(buf, offset)
    if data_dtype != "DOUBLE":
        raise ValueError(
            f"Unsupported data dtype: {data_dtype!r} "
            "(only DOUBLE is implemented; extend this parser if the "
            "project starts writing other types)"
        )

    # Read `data_count` big-endian IEEE 754 doubles. np.frombuffer with
    # dtype '>f8' is the fastest path and matches the Java writeDouble
    # format byte-for-byte.
    data_bytes = data_count * 8
    raw = np.frombuffer(buf, dtype=">f8", count=data_count, offset=offset)
    # Convert to native byte order so downstream NumPy code doesn't
    # keep dealing with the swapped view.
    data = raw.astype(np.float64, copy=True).reshape(shape)
    offset += data_bytes

    return Nd4jIndArray(shape=shape, data=data), offset


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

@dataclass
class Nd4jDataSet:
    """A parsed DataSet file — just features and labels.

    The original Java DataSet may also include masks and metadata, but
    none of the Ultimate Tic Tac Toe project files carry those, so we
    intentionally don't expose them. If a file with extra sections ever
    shows up the parser raises a clear error at the bitmask check.
    """

    features: np.ndarray
    labels: np.ndarray


def load_dataset(path: str | Path) -> Nd4jDataSet:
    """Load a legacy `.bin` DataSet file into NumPy arrays.

    Args:
        path: Path to a `.bin` file produced by the Java project's
            training data generator or concatenator.

    Returns:
        An `Nd4jDataSet` with `features` and `labels` NumPy arrays.

    Raises:
        ValueError: If the file uses a format variant the parser does
            not support (e.g. includes masks, uses FLOAT data, or has a
            different shape dtype).
    """
    path = Path(path)
    buf = path.read_bytes()

    if not buf:
        raise ValueError(f"Empty file: {path}")

    offset = 0
    bitmask, offset = _read_byte(buf, offset)

    if not (bitmask & BITMASK_FEATURES_PRESENT):
        raise ValueError(
            f"DataSet file {path} has no features block (bitmask=0x{bitmask:02x})"
        )
    if not (bitmask & BITMASK_LABELS_PRESENT):
        raise ValueError(
            f"DataSet file {path} has no labels block (bitmask=0x{bitmask:02x}); "
            "this parser does not implement LABELS_SAME_AS_FEATURES or mask-only files."
        )

    # Anything beyond features+labels is unsupported. These project
    # files should never have masks or metadata, so bail loudly if we
    # see one.
    extra_bits = bitmask & ~(BITMASK_FEATURES_PRESENT | BITMASK_LABELS_PRESENT)
    if extra_bits:
        raise ValueError(
            f"DataSet file {path} includes unsupported sections "
            f"(bitmask extras: 0x{extra_bits:02x}). Extend nd4j_loader if needed."
        )

    features_indarr, offset = _parse_indarray(buf, offset)
    labels_indarr, offset = _parse_indarray(buf, offset)

    # Sanity check: the file should be fully consumed. Any trailing
    # bytes would indicate an unhandled optional section.
    if offset != len(buf):
        raise ValueError(
            f"Trailing {len(buf) - offset} bytes after labels in {path}; "
            "the parser may be missing a format variant."
        )

    return Nd4jDataSet(
        features=features_indarr.data,
        labels=labels_indarr.data,
    )


__all__ = ["Nd4jDataSet", "Nd4jIndArray", "load_dataset"]
