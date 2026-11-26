#!/usr/bin/env python3
"""Verify that a release cdylib contains a panic at the FGN C boundary."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")).resolve() / "release/examples/libfiltergraph_fgn_panic_fixture.so"
ABI_VERSION = 8
EXECUTOR_VERSION = 1
EFAULT = 14


class Plugin(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("find_descriptor", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("n_ports", ctypes.c_uint32),
        ("ports", ctypes.c_void_p),
        ("instantiate", ctypes.c_void_p),
        ("cleanup", ctypes.c_void_p),
        ("get_port_format", ctypes.c_void_p),
        ("enum_prop_info", ctypes.c_void_p),
        ("get_prop", ctypes.c_void_p),
        ("get_prop_revision", ctypes.c_void_p),
        ("prepare_props", ctypes.c_void_p),
        ("commit_props", ctypes.c_void_p),
        ("discard_props", ctypes.c_void_p),
        ("prepare_parameter", ctypes.c_void_p),
        ("commit_parameter", ctypes.c_void_p),
        ("discard_parameter", ctypes.c_void_p),
        ("prepare_parameters", ctypes.c_void_p),
        ("adopt_parameters", ctypes.c_void_p),
        ("activate", ctypes.c_void_p),
        ("deactivate", ctypes.c_void_p),
        ("reset", ctypes.c_void_p),
        ("process", ctypes.c_void_p),
    ]


RUN_DENSE_F32 = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
RUN_LANES = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


class Executor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("n_lanes", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("run_dense_f32", RUN_DENSE_F32),
        ("run_lanes", RUN_LANES),
        ("cache_line_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blas", action="store_true", help="exercise the native BLAS/std backend")
    arguments = parser.parse_args()
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-p",
            "filtergraph-fgn",
            *(["--features", "blas"] if arguments.blas else []),
            "--example",
            "filtergraph-fgn-panic-fixture",
        ],
        cwd=ROOT,
        check=True,
    )
    library = ctypes.CDLL(LIBRARY)
    entry = library.spa_filter_graph_ndarray_plugin_get_interface
    entry.argtypes = [ctypes.c_uint32]
    entry.restype = ctypes.POINTER(Plugin)
    plugin = entry(ABI_VERSION).contents
    find_descriptor = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)(
        plugin.find_descriptor
    )
    descriptor_pointer = find_descriptor(b"panic-fixture")
    if not descriptor_pointer:
        raise RuntimeError("release fixture descriptor was not exported")
    descriptor = Descriptor.from_address(descriptor_pointer)
    instantiate = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(Executor),
        ctypes.POINTER(ctypes.c_void_p),
    )(descriptor.instantiate)
    process = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )(descriptor.process)
    cleanup = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(descriptor.cleanup)
    run_dense_f32 = RUN_DENSE_F32(lambda _data, _task: -22)
    run_lanes = RUN_LANES(lambda _data, _task: -22)
    executor = Executor(
        ctypes.sizeof(Executor),
        EXECUTOR_VERSION,
        0,
        1,
        None,
        run_dense_f32,
        run_lanes,
        64,
        0,
    )
    instance = ctypes.c_void_p()
    if instantiate(
        descriptor_pointer,
        b"{}",
        ctypes.byref(executor),
        ctypes.byref(instance),
    ) != 0:
        raise RuntimeError("release fixture did not instantiate")
    try:
        result = process(instance, None, 0, None, 0)
        if result != -EFAULT:
            raise RuntimeError(f"panic returned {result}, expected {-EFAULT}")
    finally:
        cleanup(instance)
    print("release cdylib panic was contained as -EFAULT")


if __name__ == "__main__":
    main()
