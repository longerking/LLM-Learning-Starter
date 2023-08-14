---
description: 大语言模型使用
---

# LLMs-Using

## 前言

自从OpenAI ChatGPT爆火以来，开源界 AI界刮起了大模型之风。这些国产大模型，我们当然要体验起来。本文主要从代码层面记录了如何调用体验大模型，从一个调包工程师升级成为一个调用大模型模型工程师。:poop:

## ChatGLM2-6B

ChatGLM2-6B是由清华THUDM实验室开源的中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版，本基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。

### Reference

> 1. [https://github.com/THUDM/ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
> 2. [https://huggingface.co/THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
> 3. [https://huggingface.co/THUDM/chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4)



### HardWare Requirements



<table><thead><tr><th width="149.33333333333331">量化等级</th><th width="277">编码 2048 长度的最小显存</th><th>生成 8192 长度的最小显存</th></tr></thead><tbody><tr><td>FP16 / BF16</td><td>13.1 GB</td><td>12.8 GB</td></tr><tr><td>INT8</td><td>8.2 GB</td><td>8.1 GB</td></tr><tr><td>INT4</td><td>5.5 GB</td><td>5.1 GB</td></tr></tbody></table>

这里受限于资源，我们以量化INT4的模型测试，同时INT4根据官网介绍支持CPU。



### DownLoad Models

这里直接下载Hugging Face(HF)上的int量化后的模型。HF是一个托管发布分享模型数据集的开源平台。

```shell
# 支持大文件下载
git lfs install
# 下载模型，约7GB，下载比较耗时
git clone --depth 1 https://huggingface.co/THUDM/chatglm-6b-int4
```

安装依赖

```shell
# 安装依赖，依赖源自ChatGLM2-6B-int4 HF readme
pip install protobuf transformers==4.30.2 cpm_kernels torch==2.0.1 gradio mdtex2html sentencepiece accelerate
```

### Test ChatGLM2

```python
import time
from transformers import AutoTokenizer, AutoModel
# 这里将THUDM/chatglm2-6b-int4 更换为刚模型下载存放的路径，
# 否则会自动去重新下载模型到~/.cache/目录下
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

t = time.time()
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
print(f'coast:{time.time() - t:.4f}s')
```

```
你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
睡眠对身体健康和心理健康非常重要。如果晚上睡不着，可以尝试以下一些方法:

1. 放松自己:在睡觉前数绵羊或做一些轻松的伸展运动，可以帮助放松自己。也可以尝试使用温和的瑜伽或伸展练习，缓解身体的紧张。

2. 创建一个舒适的睡眠环境:确保卧室安静、黑暗、凉爽和舒适。如果卧室中存在噪音或强光，可以考虑使用耳塞或眼罩。

3. 避免使用电子产品:在睡觉前一小时避免使用电子产品，如手机、电视或电脑。这些设备发出的蓝光会影响睡眠质量。

4. 建立一个固定的睡眠时间表:保持固定的睡眠时间表可以帮助身体建立一个正常的睡眠节律。尽量在每天相同的时间上床和起床。

5. 避免饮用刺激性饮料:在睡觉前几小时避免饮用咖啡、茶、酒或含有巧克力等刺激性饮料。

如果这些方法无效，可以考虑咨询医生以了解可能的原因。
coast:12.4483s
```

这样我们就使用官方的样例代码简单的使用ChatGLM进行推理对话了。接下来让我们可以简单的封装一下，更优美地使用ChatGLM2！
