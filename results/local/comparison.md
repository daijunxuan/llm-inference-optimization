| Configuration | Batch | P50 batch latency (ms) | P95 (ms) | Output tok/s | Peak RSS (MiB) | Device VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| HF baseline FP32 | 1 | 1954.05 | 5915.58 | 12.59 | 1996.2 | N/A |
| HF baseline FP32 | 4 | 6103.89 | 6589.97 | 20.97 | 2032.9 | N/A |
| HF dynamic INT8 | 1 | 1649.89 | 1875.23 | 18.76 | 596.9 | N/A |
| HF dynamic INT8 | 4 | 2241.68 | 2295.37 | 56.97 | 667.2 | N/A |
