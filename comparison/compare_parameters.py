#!/usr/bin/env python3
from _load import load

V4, p4 = load("deepseek_flash_v4_mini", "DeepSeekFlashV4Mini")
V41, p41 = load("deepseek_flash_v4_1_mini", "DeepSeekFlashV41Mini")
m4 = V4(p4("pc"))
cfg = p41("pc"); cfg.use_vision = False
m41 = V41(cfg)
print(f"V4-Mini   {m4.num_parameters():,}")
print(f"V4.1-Mini {m41.num_parameters():,}")
