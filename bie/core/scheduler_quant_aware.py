"""BIE Scheduler Quant-Aware — dense_layer_gb calibrado por quantização.

Baseline medida em hardware (2026-08-11, RTX 4060 8GB):
  Qwen3-8B-Q4_K_M -> 5.03 GB / 36 layers = 0.140 GB/layer a ~4.5 bits/peso.

Outras quantizações escalam pelo ratio de bits por peso em relação à Q4_K_M.
"""

from __future__ import annotations

import struct
from pathlib import Path

#: bits por peso aproximados por quantização (llama.cpp file types)
QUANT_BITS: dict[str, float] = {
    "Q2_K": 2.6,
    "Q3_K_S": 3.25,
    "Q3_K_M": 3.5,
    "Q3_K_L": 3.75,
    "Q4_0": 4.25,
    "Q4_K_S": 4.25,
    "Q4_K_M": 4.5,
    "Q4_K_L": 4.75,
    "Q5_K_S": 5.25,
    "Q5_K_M": 5.5,
    "Q5_K_L": 5.75,
    "Q6_K": 6.5,
    "Q8_0": 8.0,
    "F16": 16.0,
}

#: medição real de referência (profiler em hardware, 2026-08-11)
MEASURED_REF = {"quant": "Q4_K_M", "bits": QUANT_BITS["Q4_K_M"], "dense_layer_gb": 0.140}


def quant_bits(quant: str) -> float:
    if quant not in QUANT_BITS:
        raise ValueError(f"Quantização não suportada: {quant}. Suportadas: {sorted(QUANT_BITS)}")
    return QUANT_BITS[quant]


def derive_dense_layer_gb(model_size_gb: float, num_layers: int, quant: str) -> float:
    """Estima dense_layer_gb para a quantização, escalando a medição Q4_K_M.

    dense_layer_gb = (model_size_gb / num_layers) * (bits(quant) / bits(Q4_K_M))
    """
    if model_size_gb <= 0 or num_layers <= 0:
        raise ValueError("model_size_gb e num_layers devem ser positivos")
    scale = quant_bits(quant) / MEASURED_REF["bits"]
    return round((model_size_gb / num_layers) * scale, 4)


def vram_estimate_gb(dense_layer_gb: float, n_gpu_layers: int, kv_cache_gb: float = 0.0) -> float:
    """Estimativa simples de VRAM para o offload planejado."""
    return round(dense_layer_gb * n_gpu_layers + kv_cache_gb, 2)


def gguf_metadata(gguf_path: str | Path) -> dict:
    """Lê metadados reais do GGUF (architecture, block_count, file_type).

    Usa o leitor do llama.cpp quando disponível; fallback para parser próprio.
    """
    path = Path(gguf_path)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF não encontrado: {path}")
    size_gb = max(0.001, round(path.stat().st_size / (1024**3), 3))
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=str(path), n_gpu_layers=0, verbose=False)
        md = dict(getattr(llm, "metadata", None) or {})
        arch = str(md.get("general.architecture", ""))
        block_count = int(md.get(f"{arch}.block_count", 0))
        file_type = int(md.get("general.file_type", 0))
        del llm
    except (ImportError, OSError, RuntimeError, ValueError):
        parsed = _gguf_metadata_manual(path)
        arch = parsed.get("architecture", "")
        block_count = parsed.get("block_count", 0)
        file_type = parsed.get("file_type", 0)
    return {
        "path": str(path),
        "architecture": arch,
        "block_count": block_count,
        "file_type": file_type,
        "size_gb": size_gb,
    }


def derive_from_gguf(gguf_path: str | Path, quant: str | None = None) -> dict:
    """Deriva dense_layer_gb do GGUF real + quantização (default: do nome do arquivo)."""
    meta = gguf_metadata(gguf_path)
    if meta["block_count"] <= 0:
        raise ValueError(f"block_count não lido para {gguf_path} (llama_cpp necessário)")
    if quant is None:
        quant = _quant_from_filename(Path(gguf_path).name, meta["file_type"])
    dense = derive_dense_layer_gb(meta["size_gb"], meta["block_count"], quant)
    return {
        "quant": quant,
        "dense_layer_gb": dense,
        "vram_6_layers_gb": vram_estimate_gb(dense, 6),
        "vram_22_layers_gb": vram_estimate_gb(dense, 22),
        **meta,
    }


def _gguf_metadata_manual(path: Path) -> dict:
    """Parser próprio de metadados GGUF (fallback sem carregar o arquivo inteiro na memória)."""
    with open(path, "rb") as f:
        raw = f.read(16 * 1024 * 1024)  # 16 MB is plenty for KV metadata headers
    if raw[:4] != b"GGUF":
        return {"architecture": "", "block_count": 0, "file_type": 0}
    n_kv = struct.unpack("<Q", raw[16:24])[0]
    offset = 24
    kv: dict[str, object] = {}
    for _ in range(n_kv):
        key, offset = _read_gguf_key(raw, offset)
        try:
            value, offset = _read_gguf_value(raw, offset)
        except (ValueError, struct.error, IndexError):
            break
        kv[key] = value
    arch = str(kv.get("general.architecture", ""))
    block_count = int(kv.get(f"{arch}.block_count", 0)) if arch else 0
    file_type = int(kv.get("general.file_type", 0))

    # Robust fast scan if key was located after complex arrays
    if block_count == 0:
        import re

        matches = re.findall(rb"([a-zA-Z0-9_\.]+)\.block_count", raw)
        for cand_arch in matches:
            full_key = cand_arch + b".block_count"
            idx = raw.find(full_key)
            if idx != -1:
                val_offset = idx + len(full_key)
                if val_offset + 8 <= len(raw):
                    vtype = struct.unpack("<I", raw[val_offset : val_offset + 4])[0]
                    if vtype in (4, 5):  # u32, i32
                        block_count = struct.unpack("<I", raw[val_offset + 4 : val_offset + 8])[0]
                        if not arch:
                            arch = cand_arch.decode("utf-8", errors="replace")
                        break

    return {
        "architecture": arch,
        "block_count": block_count,
        "file_type": file_type,
    }


def _quant_from_filename(name: str, file_type: int) -> str:
    stem = Path(name).stem.upper()
    for q in sorted(QUANT_BITS, key=len, reverse=True):
        if q in stem:
            return q
    family = {10: "Q2_K", 11: "Q3_K_M", 12: "Q4_K_M", 13: "Q5_K_M", 14: "Q6_K", 7: "Q8_0", 2: "F16", 1: "F16"}.get(file_type, "Q4_K_M")
    return family


_GGUF_TYPE_MAP = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
    6: "f32", 7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64",
    13: "f16", 14: "bf16",
}


def _read_gguf_key(data: bytes, offset: int) -> tuple[str, int]:
    n, offset = _read_gguf_int(data, offset, 8)
    key = data[offset : offset + n].decode("utf-8", errors="replace")
    return key, offset + n


def _read_gguf_value(data: bytes, offset: int) -> tuple[object, int]:
    vtype, offset = _read_gguf_int(data, offset, 4)
    vtype = _GGUF_TYPE_MAP.get(vtype, "unknown")
    if vtype == "string":
        return _read_gguf_key(data, offset)
    if vtype == "array":
        _etype, offset = _read_gguf_int(data, offset, 4)
        n, offset = _read_gguf_int(data, offset, 8)
        items: list[object] = []
        for _ in range(n):
            item, offset = _read_gguf_value(data, offset)
            items.append(item)
        return items, offset
    if vtype in ("u32", "i32"):
        return _read_gguf_int(data, offset, 4)
    if vtype in ("u64", "i64"):
        return _read_gguf_int(data, offset, 8)
    if vtype in ("u8", "i8"):
        return data[offset], offset + 1
    if vtype in ("u16", "i16", "f16"):
        return struct.unpack("<H", data[offset : offset + 2])[0], offset + 2
    if vtype == "f32":
        return struct.unpack("<f", data[offset : offset + 4])[0], offset + 4
    if vtype == "f64":
        return struct.unpack("<d", data[offset : offset + 8])[0], offset + 8
    if vtype == "bool":
        return data[offset] != 0, offset + 1
    raise ValueError(f"tipo GGUF não suportado: {vtype}")


def _read_gguf_int(data: bytes, offset: int, size: int) -> tuple[int, int]:
    return struct.unpack("<Q" if size == 8 else "<I", data[offset : offset + size])[0], offset + size