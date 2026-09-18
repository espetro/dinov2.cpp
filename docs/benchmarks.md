# Benchmarks

All numbers below are 100-run averages on an Intel Core i9-14900HX (24 cores, 32 threads) with 24 threads. Full methodology and additional numbers: [release post](https://alexlavaee.me/projects/dinov2cpp/).

Headline: dinov2.cpp is up to **3x faster than native PyTorch inference on CPU** for the small model and roughly **1.5-2x faster** for the rest, with substantially lower memory use.

## DINOv2 inference vs PyTorch

### With register tokens

Models: `dinov2-with-registers-{size}-imagenet1k-1-layer`

| Model | Max Mem (PyTorch) | Max Mem | Speed (PyTorch) | Speed |
|:-----:|:-----------------:|:-------:|:---------------:|:-----:|
| small | ~457 MB | **~109 MB** | 297 ms | **64 ms** |
| base | ~720 MB | **~367 MB** | 436 ms | **200 ms** |
| large | ~1.57 GB | **~1.2 GB** | 1331 ms | **597 ms** |
| giant | ~4.8 GB | **~4.4 GB** | 4472 ms | **1995 ms** |

### Without register tokens

Models: `dinov2-{size}-imagenet1k-1-layer`

| Model | Max Mem (PyTorch) | Max Mem | Speed (PyTorch) | Speed |
|:-----:|:-----------------:|:-------:|:---------------:|:-----:|
| small | ~455 MB | **~110 MB** | 181 ms | **62 ms** |
| base | ~720 MB | **~367 MB** | 462 ms | **197 ms** |
| large | ~1.55 GB | **~1.2 GB** | 1288 ms | **600 ms** |
| giant | ~4.8 GB | **~4.4 GB** | 4384 ms | **1969 ms** |

## Quantized models

Benchmarks for quantization types on the same machine, 100 runs each.

### With register tokens

| Model | Quantization | Speed (ms) | Mem (MB) |
|:-----:|:------:|:----------:|:--------:|
| small | q4_0 | 52 | 49 |
| small | q4_1 | 50 | 52 |
| small | q5_0 | 59 | 54 |
| small | q5_1 | 57 | 57 |
| small | q8_0 | 51 | 70 |
| base | q4_0 | 136 | 129 |
| base | q4_1 | 133 | 139 |
| base | q5_0 | 164 | 150 |
| base | q5_1 | 158 | 160 |
| base | q8_0 | 124 | 211 |
| large | q4_0 | 395 | 371 |
| large | q4_1 | 395 | 407 |
| large | q5_0 | 493 | 443 |
| large | q5_1 | 490 | 480 |
| large | q8_0 | 353 | 661 |
| giant | q4_0 | 1275 | 1281 |
| giant | q4_1 | 1261 | 1417 |
| giant | q5_0 | 1615 | 1552 |
| giant | q5_1 | 1583 | 1687 |
| giant | q8_0 | 1065 | 2364 |

### Without register tokens

| Model | Quantization | Speed (ms) | Mem (MB) |
|:-----:|:------:|:----------:|:--------:|
| small | q4_0 | 46 | 49 |
| small | q4_1 | 48 | 51 |
| small | q5_0 | 63 | 54 |
| small | q5_1 | 58 | 57 |
| small | q8_0 | 50 | 70 |
| base | q4_0 | 141 | 129 |
| base | q4_1 | 135 | 140 |
| base | q5_0 | 162 | 150 |
| base | q5_1 | 161 | 160 |
| base | q8_0 | 125 | 212 |
| large | q4_0 | 389 | 371 |
| large | q4_1 | 382 | 407 |
| large | q5_0 | 497 | 444 |
| large | q5_1 | 478 | 480 |
| large | q8_0 | 348 | 661 |
| giant | q4_0 | 1268 | 1281 |
| giant | q4_1 | 1248 | 1417 |
| giant | q5_0 | 1625 | 1553 |
| giant | q5_1 | 1576 | 1688 |
| giant | q8_0 | 1059 | 2364 |

## Run your own

Benchmark scripts and instructions: [build.md#benchmarks](build.md#benchmarks).
