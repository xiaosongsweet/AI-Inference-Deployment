
# AI 推理部署实验记录

## 实验 04.1：PyTorch 与 ONNX CPU 性能对比

### 1. 实验目的

比较 PyTorch 和 ONNX Runtime 在 CPU 环境下运行同一个 CIFAR-10 CNN 模型时的推理性能。

### 2. 实验环境

- 模型：CIFAR10CNN
- 数据集：CIFAR-10
- 运行设备：CPU
- Batch Size：1
- 预热次数：30
- 正式测量次数：20
- ONNX Runtime：1.23.2
- ONNX：1.23.2

### 3. 实验结果

| 性能指标 | PyTorch CPU | ONNX Runtime CPU |
|---|---:|---:|
| 平均延迟 | 0.6723 ms | 0.2011 ms |
| 吞吐量 | 1487.45 images/s | 4971.54 images/s |

ONNX 相对 PyTorch 加速比：3.34x

### 4. 实验分析

在本次 Batch Size = 1 的测试中：

1. ONNX Runtime 的平均推理延迟低于 PyTorch。
2. ONNX Runtime 的吞吐量高于 PyTorch。
3. 本次测试中，ONNX Runtime 的推理加速比约为 3.34 倍。

### 5. 实验结论

在当前模型、CPU 环境和 Batch Size = 1 的测试条件下，ONNX Runtime 表现出更好的推理性能。

但本次正式测量次数只有 20 次，样本量较少，后续需要增加测量次数，并测试不同 Batch Size，进一步验证结果。

### 6. 后续实验计划

- 增加正式测量次数至 200 次。
- 测试 Batch Size = 1、4、16、32、64。
- 比较不同 Batch Size 下的延迟和吞吐量。
- 分析 CPU 线程数对推理性能的影响。


## 实验 04.1 复测：正式测量 200 次

### 实验配置

- Batch Size：1
- 预热次数：30
- 正式测量次数：200
- 运行设备：CPU

### 实验结果

| 性能指标 | PyTorch CPU | ONNX Runtime CPU |
|---|---:|---:|
| 平均延迟 | 0.6811 ms | 0.2003 ms |
| 吞吐量 | 1468.31 images/s | 4991.80 images/s |

ONNX 相对 PyTorch 加速比：3.40x

### 实验分析

增加正式测量次数后，ONNX Runtime 仍然表现出更低的推理延迟。

与 20 次测量相比，两次实验的结果较为接近。

### 实验结论

在当前 CIFAR10CNN、CPU 单线程和 Batch Size=1 的测试条件下，
ONNX Runtime 的推理性能优于 PyTorch。

后续需要进一步测试不同 Batch Size 和 CPU 线程数。


## 实验 04.2：不同 Batch Size 的推理性能对比

### 1. 实验目的

研究不同 Batch Size 对 PyTorch CPU 和 ONNX Runtime CPU
推理性能的影响，分析批量推理对吞吐量和延迟的作用。

### 2. 实验配置

- 模型：CIFAR10 CNN
- 运行设备：CPU
- CPU 线程数：1
- Batch Size：1、4、16、32、64
- 预热次数：30
- 正式测量次数：200
- ONNX Runtime：CPUExecutionProvider

### 3. 实验结果

| Batch Size | PyTorch 延迟(ms) | ONNX 延迟(ms) | PyTorch 吞吐量(images/s) | ONNX 吞吐量(images/s) | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7248 | 0.2099 | 1379.61 | 4764.06 | 3.45x |
| 4 | 2.3130 | 0.6923 | 1729.39 | 5777.64 | 3.34x |
| 16 | 8.3071 | 2.6892 | 1926.07 | 5949.81 | 3.09x |
| 32 | 16.1565 | 5.5184 | 1980.62 | 5798.76 | 2.93x |
| 64 | 32.9381 | 10.6275 | 1943.04 | 6022.09 | 3.10x |

### 4. 实验分析

1. 随着 Batch Size 增大，PyTorch 和 ONNX Runtime 的吞吐量总体提高。
2. PyTorch 在 Batch Size=32 时取得最高吞吐量，为 1980.62 images/s。
3. ONNX Runtime 在 Batch Size=64 时取得最高吞吐量，为 6022.09 images/s。
4. ONNX Runtime 在所有测试条件下均快于 PyTorch，加速比为 2.93x～3.45x。
5. 单张图片平均分摊延迟随 Batch Size 增大总体下降，但 Batch Size=64 时 PyTorch 略有回升。

### 5. 实验结论

在当前模型、CPU 和单线程测试条件下，批量推理能够提高整体吞吐量。

ONNX Runtime 在本次测试中表现出更高的吞吐量。

不同 Batch Size 对两种推理引擎的影响并不完全相同，
因此实际部署时需要根据吞吐量、响应延迟和内存占用选择合适的 Batch Size。

### 6. 实验局限性

本次实验采用平均延迟作为主要指标，尚未统计 P50、P95 等延迟分位数。
每种 Batch Size 只进行了一轮正式测试，后续需要多轮重复实验验证稳定性。



## 实验 04.3：Benchmark 可靠性优化与延迟分布分析

### 1. 实验目的

在实验 04.2 的基础上，进一步提高 Benchmark 测量的可靠性。

本次实验主要改进：

1. 将单轮测量扩展为三轮独立测量。
2. 每轮正式测量前执行 30 次预热。
3. 每轮正式测量 200 次，每种 Batch Size 累计获得 600 个延迟样本。
4. 增加 Mean、P50、P95 延迟统计指标。
5. 通过交替调整 PyTorch 和 ONNX Runtime 的测量顺序，减少固定执行顺序带来的影响。

### 2. 实验环境与参数

| 参数 | 设置 |
|---|---|
| 模型 | CIFAR-10 CNN |
| 推理框架 | PyTorch |
| 部署格式 | ONNX |
| 推理引擎 | ONNX Runtime |
| 设备 | CPU |
| CPU 线程数 | 1 |
| Batch Size | 1、4、16、32、64 |
| Warmup | 30 次/轮 |
| Repeat | 200 次/轮 |
| Rounds | 3 轮 |
| 正式测量样本 | 600 个/引擎/Batch Size |
| 延迟统计 | Mean、P50、P95 |

### 3. 实验结果

| Batch Size | PyTorch Mean (ms) | PyTorch P50 (ms) | PyTorch P95 (ms) | ONNX Mean (ms) | ONNX P50 (ms) | ONNX P95 (ms) | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7630 | 0.6627 | 1.3918 | 0.2218 | 0.1895 | 0.4828 | 3.44× |
| 4 | 2.8334 | 2.5324 | 4.4234 | 0.8684 | 0.7292 | 1.6921 | 3.26× |
| 16 | 9.2079 | 8.7020 | 12.6894 | 2.9441 | 2.7365 | 4.5276 | 3.13× |
| 32 | 17.7112 | 16.5421 | 24.7413 | 5.9287 | 5.3403 | 9.7837 | 2.99× |
| 64 | 34.4084 | 33.7387 | 39.6106 | 10.9183 | 10.6234 | 12.3128 | 3.15× |

### 4. 实验分析

1. 在所有测试条件下，ONNX Runtime 的平均推理延迟均低于 PyTorch。
2. ONNX Runtime 的加速比在 2.99×～3.44× 之间，表现出较稳定的性能优势。
3. 两种推理引擎的 P95 均高于 P50，说明部分测量样本存在较高的尾部延迟。
4. 随着 Batch Size 增大，两种推理引擎的总吞吐量总体呈上升趋势，但并非严格单调。
5. ONNX Runtime 在 Batch Size=64 时取得最高吞吐量，为 5861.72 images/s。
6. Batch Size 增大时，批次总延迟也随之增加。因此，实际部署时不能只关注吞吐量，还需要综合考虑请求延迟。

