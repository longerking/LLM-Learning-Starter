---
description: 大语言模型使用
---

# ChatGLM-Using

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

## FastAPI Deploy ChatGLM2

上面我们已经学会了下载模型，然后进行简单的模型对话推理。接下来使用我们使用FastAPI可以将模型部署到服务器上，以接口方式进行对话。注意这里的API仅仅是简单封装，不同于FastChat类OpenAPI提供的接口。

> FastAPI: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

```python
# -*-coding:utf-8-*-
'''
File Name:chatglm2-6b-stream-api.py
Author:LongerKing
Time:2023/8/15 13:33
'''

import os
import sys
import json
import torch
import uvicorn
import logging
import argparse
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import ServerSentEvent, EventSourceResponse


def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = getLogger('ChatGLM', 'chatlog.log')

MAX_HISTORY = 3


class ChatGLM():
    def __init__(self) -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/bigdata/wonders/wanglang/models/chatglm2-6b-cmeee_eie001", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/home/bigdata/wonders/wanglang/models/chatglm2-6b-cmeee_eie001",
                                               trust_remote_code=True).quantize(8).cuda()
        self.model.eval()
        logger.info("Model initialization finished.")

    def clear(self) -> None:
        if torch.cuda.is_available():
            with torch.cuda.device(f"cuda:{args.device}"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def answer(self, query: str, history):
        response, history = self.model.chat(self.tokenizer, query, history=history)
        history = [list(h) for h in history]
        return response, history

    def stream(self, query, history):
        if query is None or history is None:
            yield {"query": "", "response": "", "history": [], "finished": True}
        size = 0
        response = ""
        for response, history in self.model.stream_chat(self.tokenizer, query, history):
            this_response = response[size:]
            history = [list(h) for h in history]
            size = len(response)
            yield {"delta": this_response, "response": response, "finished": False}
        logger.info("Answer - {}".format(response))
        yield {"query": query, "delta": "[EOS]", "response": response, "history": history, "finished": True}


def start_server(http_address: str, port: int, gpu_id: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    bot = ChatGLM()

    app = FastAPI()
    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"],
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"]
                       )

    @app.get("/")
    def index():
        return {'message': 'started', 'success': True}

    @app.post("/chat")
    async def answer_question(arg_dict: dict):
        result = {"query": "", "response": "", "success": False}
        try:
            text = arg_dict["query"]
            ori_history = arg_dict["history"]
            logger.info("Query - {}".format(text))
            if len(ori_history) > 0:
                logger.info("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            response, history = bot.answer(text, history)
            logger.info("Answer - {}".format(response))
            ori_history.append((text, response))
            result = {"query": text, "response": response,
                      "history": ori_history, "success": True}
        except Exception as e:
            logger.error(f"error: {e}")
        return result

    @app.post("/stream")
    def answer_question_stream(arg_dict: dict):
        def decorate(generator):
            for item in generator:
                yield ServerSentEvent(json.dumps(item, ensure_ascii=False), event='delta')

        try:
            text = arg_dict["query"]
            ori_history = arg_dict["history"]
            logger.info("Query - {}".format(text))
            if len(ori_history) > 0:
                logger.info("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            return EventSourceResponse(decorate(bot.stream(text, history)))
        except Exception as e:
            logger.error(f"error: {e}")
            return EventSourceResponse(decorate(bot.stream(None, None)))

    @app.get("/free_gc")
    def free_gpu_cache():
        try:
            bot.clear()
            return {"success": True}
        except Exception as e:
            logger.error(f"error: {e}")
            return {"success": False}

    logger.info("starting server...")
    uvicorn.run(app=app, host=http_address, port=port, workers=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stream API Service for ChatGLM2-6B')
    parser.add_argument('--device', '-d', help='device，-1 means cpu, other means gpu ids', default='0')
    parser.add_argument('--host', '-H', help='host to listen', default='0.0.0.0')
    parser.add_argument('--port', '-P', help='port of this service', default=8000)
    args = parser.parse_args()
    start_server(args.host, int(args.port), args.device)
```
