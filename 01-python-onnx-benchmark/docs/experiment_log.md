
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
