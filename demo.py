from openai import OpenAI

client = OpenAI(base_url="http://localhost:10103/v1", api_key="EMPTY") # 这里的10103需要对齐SGLang Server的port

response = client.chat.completions.create(
    model="DeepSeek-V2-Lite-Chat:lora0",
    messages=[{"role": "user", "content": "你是谁"}],
    temperature=0,
    top_p=1,
    max_tokens=128,
)
print(response.choices[0].message.content)

# 对比基座（可选）
response = client.chat.completions.create(
    model="DeepSeek-V2-Lite-Chat",
    messages=[{"role": "user", "content": "你是谁"}],
    temperature=0,
    top_p=1,
    max_tokens=128,
)
print(response.choices[0].message.content)
