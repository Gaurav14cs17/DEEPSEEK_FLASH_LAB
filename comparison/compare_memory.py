#!/usr/bin/env python3
"""Parameter + KV accounting for both Minis."""

from _load import load

from deepseek_flash_v4_mini.optimization.kv_cache import estimate_kv as est4
from deepseek_flash_v4_mini.optimization.memory import param_bytes
from deepseek_flash_v4_1_mini.optimization.kv_cache import estimate_kv as est41
from deepseek_flash_v4_1_mini.optimization.memory import param_bytes as pb41

V4, p4 = load("deepseek_flash_v4_mini", "DeepSeekFlashV4Mini")
m4 = V4(p4("pc"))
print("V4", param_bytes(m4), est4(m4.cfg, 64).summary())

V41, p41 = load("deepseek_flash_v4_1_mini", "DeepSeekFlashV41Mini")
cfg = p41("pc")
cfg.use_vision = False
m41 = V41(cfg)
print("V4.1", pb41(m41), est41(m41.cfg, 64).summary())
