---
description: 利用FastChat部署ChatGLM
---

# ChatGLM-Deploy-FastChat

Reference

> https://github.com/lm-sys/FastChathttps://github.com/lm-sys/FastChat/blob/main/docs/openai\_api.md

\
`Note`： 一些screen窗口会话知识预备

```sh
screen -S fs-glm-wkr # 创建一个名为fs-glm-wkr的会话
# ctrl+shift+d # 退出当前会话窗口，但会话内容保持后台
screen -r fs-glm-wkr # 连接到 fs-glm-wkr 会话窗口
# ctrl+d # 退出会话窗口，并停止整个窗口内程序
screen -ls # 查看所有会话窗口
screen -D -r fs-glm-wkr # 强制重连已经Attacted状态的窗口
```

\


### Install FastChat

使用依赖包安装或者使用源码安装，二选一即可。如果FastChat还不支持的模型，建议使用源码安装，再源码集成支持模型后在源码安装依赖包。 `Note`: FastChat支持对接chatglm是因为源码中，已经做了这部分的配置集成。这里的测试使用pip安装。

```sh
pip3 install fschat
# or
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .
```

\


### Load Models

服务器部署模型，并提供l类似gpt 的 openai 接口方式访问。

1. #### Use serve.cli testing model

cli运行脚本如下

```sh
python3 -m fastchat.serve.cli \
        --model-path /mnt/models/chatglm2-6b
```

如图，会在终端生成如下交互问答

````prolog
(chatglm)longer@codeWL$ python3 -m fastchat.serve.cli --model-path /mnt/models/chatglm2-6b
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:13<00:00,  1.18it/s]
问: 你是谁？
答: 我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型开发的。我的任务是针对用户的问题和要求提供适当的答复和支持。
问: 如何使用fastchat
答: Fastchat 是一个基于 Python 的聊天应用程序，可以使用它进行文本或语音聊天。要使用 Fastchat，您需要首先安装它。在终端中输入以下命令来安装 Fastchat：
```
pip install fastchat
```
安装完成后，您可以运行以下命令启动 Fastchat：
```
fastchat
```
然后，您可以通过输入 "你好" 来启动 Fastchat 的语音聊天功能。
```
fastchat 你好
```
在 Fastchat 中，您可以向对方发送文本消息或语音消息。在发送文本消息时，您可以使用 emoji 表情符号和表情符来使消息更加生动。例如，要发送一个带有开心表情的表情符号，您可以使用以下代码：
```
fastchat :(
```
要发送一个带有哭泣表情的表情符号，您可以使用以下代码：
```
fastchat :'(
```
此外，Fastchat 还支持发送语音消息。要发送语音消息，请首先请确保您的计算机已安装了浏览器，并将 Fastchat 的语音消息 URL 复制到浏览器中。然后，您可以使用以下代码发送语音消息：
```
fastchat :(4437912：您想说什么
```
这将在您的计算机上打开一个浏览器，并在 Fastchat 中显示您输入的语音消息。
问:
````

\


2. #### serve.controller for RESTful API Server

**First Lunch a fastchat.serve.controller。**

```prolog
# screen -S fs-glm-ctl
python3 -m fastchat.serve.controller

# 输出如下
(chatglm)longer@codeWL$ python3 -m fastchat.serve.controller
2023-08-01 07:15:20 | INFO | controller | args: Namespace(host='localhost', port=21001, dispatch_method='shortest_queue')
2023-08-01 07:15:20 | ERROR | stderr | INFO:     Started server process [18496]
2023-08-01 07:15:20 | ERROR | stderr | INFO:     Waiting for application startup.
2023-08-01 07:15:20 | ERROR | stderr | INFO:     Application startup complete.
2023-08-01 07:15:20 | ERROR | stderr | INFO:     Uvicorn running on http://localhost:21001 (Press CTRL+C to quit)
```

**Then, launch the model worker(s)**

```sh
#  screen -S fs-glm-wkr
python3 -m fastchat.serve.model_worker --model-path /mnt/models/chatglm2-6b

```

\


**Finally, launch the RESTful API server**

```shell
#  screen -S fs-glm-api
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
# localhost 建议使用局域网或者公网地址，这是对外的api接口
```

\
`Note`: 以上三个步骤都是必须按照步骤全部执行。可以使用screen session会话依次后台执行这些服务。\


**Launch the Gradio web server**

```sh
# 使用fastchat gradio页面来查看加载的模型
# screen -S fs-glm-grd
python3 -m fastchat.serve.gradio_web_server
# ... 
# Running on local URL:  http://0.0.0.0:7860
# ...
```

Grade web Page：上面的model\_worker只选择了一个，所以这里只显示已经加载的模型。如果要支持多个模型可以看下文的**Running Mutiple**，当然这个也需要显存支持。

### Test APIs

#### Interact with mode（OpenAI方式）

> http://hostserver:8000/v1/

运行 `chatglm.py` 脚本会得到如下的ResponseText

```sh
(chatai) PS E:\TestGLM\src> python chatglm.py
你好，作为一名人工智能助手，我无法感受到情感，但我可以提供帮助。 请问有什么问题我可以解答吗？
Hello! My name is Noxix. I am an AI chatbot designed to assist you with any questions or tasks you may have. How can I help you today?
```

`chatglm.py` 内容如下

```python
# pip install --upgrade openai
import openai
# to get proper authentication, make sure to use a valid key that's listed in
# the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
openai.api_key = "EMPTY"
openai.api_base = "http://yourserverip:8000/v1"

model = "chatglm2-6b"  # 模型文件地址，文件夹名字
prompt = "你好"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)

#########################类似opena的代码##################################
import openai
def create(**args):
    openai.api_key = "EMPTY"
    openai.api_base = "http://hostserver:8000/v1"

    try:
        result = openai.ChatCompletion.create(**args)
    except openai.error.RateLimitError:
        result = create(**args)
    
    return result

def chat(mess):
    responde = create(
        model="chatglm2-6b",
        messages=mess
    )

    res = responde['choices'][0]['message']['content']
    return res
```

#### List Models

> http://hostserver:8000/v1/models

Using CMD `curl http://hostserver:8000/v1/models`， then

```json
// curl http://hostserver:8000/v1/models
{
    "object": "list",
    "data": [
        {
            "id": "chatglm2-6b",
            "object": "model",
            "created": 1690876019,
            "owned_by": "fastchat",
            "root": "chatglm2-6b",
            "parent": null,
            "permission": [
                {
                    "id": "modelperm-37FkwejmTx3qArXrHS5uSv",
                    "object": "model_permission",
                    "created": 1690876019,
                    "allow_create_engine": false,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": true,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ]
        }
    ]
}
```

#### Chat Completions

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{    "model": "vicuna-7b-v1.3",    "messages": [{"role": "user", "content": "Hello! What is your name?"}]  }'
```

#### Text Completions

```sh
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{    "model": "vicuna-7b-v1.3",    "prompt": "Once upon a time",    "max_tokens": 41,    "temperature": 0.5  }'
```

#### Embeddings:

```sh
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{    "model": "vicuna-7b-v1.3",    "input": "Hello world!"  }'
```



### Running Multiple（TODO TEST 多模型加载）

If you want to run multiple models on the same machine and in the same process, you can replace the `model_worker` step above with a multi model variant:

```sh
# screen -r fs-glm-wkr

python3 -m fastchat.serve.multi_model_worker \
    --model-path /mnt/models/chatglm2-6b-cmeee_eie001 \
    --model-names chatglm2-6b-cmeee_eie001
    --model-path /mnt/models/chatglm2-6b \
    --model-names chatglm2-6b 
```

`Note` 同时加载多个模型，设置模型名称路径即可，需要显存支持。这里尝试运行了两个ChatGLM以及微调ChatGLM, 显然24G的显存不支持同时运行两个。故之后的测试步骤暂不摘录。\
接下来就愉快的使用openai去对接chatglm应用吧。
