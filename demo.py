from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# 方式1：使用 chat.completions API（推荐，因为是 Chat 模型）
response = client.chat.completions.create(
    model="DeepSeek:klora",  # 关键：用冒号指定 LoRA adapter
    messages=[
        {"role": "user", "content": "你好"}
    ],
    max_tokens=256,
)
print(response.choices[0].message.content)

# 方式2：如果要用 completions API
response = client.completions.create(
    model="DeepSeek:klora",  # 同样需要 :klora
    prompt="你好",
    max_tokens=256,
)
print(response.choices[0].text)