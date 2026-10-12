"""Paged Attention Memory Block Allocator for BIE.

Models non-contiguous KV block allocation (vLLM style) to eliminate memory fragmentation
and enable zero-copy dynamic sequence expansion on consumer hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class PhysicalBlock:
    block_id: int
    tokens_capacity: int = 16
    tokens_used: int = 0

    @property
    def is_full(self) -> bool:
        return self.tokens_used >= self.tokens_capacity


class PagedCacheAllocator:
    """Manages virtual-to-physical block tables for KV cache pages."""

    def __init__(
        self,
        total_blocks: int = 512,
        block_size: int = 16,  # tokens per block
        bytes_per_token: float = 64.0,  # e.g., quantized KV bytes
    ) -> None:
        self.total_blocks = max(1, total_blocks)
        self.block_size = max(1, block_size)
        self.bytes_per_token = bytes_per_token

        # Pool of physical block IDs
        self.free_blocks: list[int] = list(range(self.total_blocks))
        # Map: seq_id -> list of block_ids
        self.block_tables: dict[str, list[int]] = {}
        # Map: block_id -> PhysicalBlock object
        self.blocks: dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, tokens_capacity=block_size)
            for i in range(self.total_blocks)
        }

    def allocate(self, seq_id: str, num_tokens: int) -> list[int]:
        """Allocate required blocks for a sequence with num_tokens."""
        blocks_needed = math.ceil(num_tokens / self.block_size)
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError(
                f"Out of Paged Cache blocks: requested {blocks_needed}, free {len(self.free_blocks)}"
            )

        allocated: list[int] = []
        for _ in range(blocks_needed):
            b_id = self.free_blocks.pop(0)
            allocated.append(b_id)

        # Update tokens_used on blocks
        remaining = num_tokens
        for b_id in allocated:
            used = min(remaining, self.block_size)
            self.blocks[b_id].tokens_used = used
            remaining -= used

        self.block_tables[seq_id] = allocated
        return allocated

    def append_tokens(self, seq_id: str, num_tokens: int = 1) -> bool:
        """Append tokens to an active sequence, allocating new blocks if needed."""
        if seq_id not in self.block_tables:
            return False

        table = self.block_tables[seq_id]
        last_block = self.blocks[table[-1]] if table else None

        remaining = num_tokens
        while remaining > 0:
            if last_block is not None and not last_block.is_full:
                can_take = last_block.tokens_capacity - last_block.tokens_used
                take = min(remaining, can_take)
                last_block.tokens_used += take
                remaining -= take
            else:
                if not self.free_blocks:
                    return False
                new_id = self.free_blocks.pop(0)
                last_block = self.blocks[new_id]
                last_block.tokens_used = 0
                table.append(new_id)

        return True

    def free(self, seq_id: str) -> None:
        """Release all physical blocks allocated to seq_id back to pool."""
        if seq_id in self.block_tables:
            allocated = self.block_tables.pop(seq_id)
            for b_id in allocated:
                self.blocks[b_id].tokens_used = 0
                self.free_blocks.append(b_id)

    def stats(self) -> dict[str, Any]:
        """Compute allocation efficiency and memory footprint."""
        used_blocks_count = self.total_blocks - len(self.free_blocks)
        total_tokens_stored = sum(
            self.blocks[b_id].tokens_used
            for table in self.block_tables.values()
            for b_id in table
        )
        total_capacity_tokens = used_blocks_count * self.block_size
        efficiency = (
            round((total_tokens_stored / total_capacity_tokens) * 100.0, 1)
            if total_capacity_tokens > 0
            else 100.0
        )
        total_vram_mb = round((self.total_blocks * self.block_size * self.bytes_per_token) / (1024 * 1024), 2)
        active_vram_mb = round((used_blocks_count * self.block_size * self.bytes_per_token) / (1024 * 1024), 2)

        return {
            "total_blocks": self.total_blocks,
            "free_blocks": len(self.free_blocks),
            "allocated_blocks": used_blocks_count,
            "active_sequences": len(self.block_tables),
            "total_tokens_stored": total_tokens_stored,
            "block_utilization_pct": efficiency,
            "total_paged_vram_mb": total_vram_mb,
            "active_paged_vram_mb": active_vram_mb,
        }
