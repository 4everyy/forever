from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# 1. 初始化 API 服务
app = FastAPI()

# 2. 配置 OpenAI (你需要填入你的 Key，或者配置环境变量)
# 如果你用的是国内的中转 Key，记得修改 base_url
client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # 替换成你的 OpenAI Key
    # base_url="https://api.openai-proxy.com/v1" # 如果用国内中转，取消注释这行
)

# 3. 定义输入的数据格式 (就像你看到的 JSON)
class UserInput(BaseModel):
    text: str

# 4. 定义 Agent 的核心逻辑
@app.post("/api/analyze_sentiment")
async def analyze_sentiment(input_data: UserInput):
    try:
        # 设定 Agent 的人设 (System Prompt)
        system_prompt = "你是一个情感分析专家。请分析用户输入的文本，只返回以下三个词之一：【正面】、【负面】、【中性】。不要说废话。"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # 或者 gpt-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_data.text}
            ]
        )
        
        # 获取 AI 的回答
        result = response.choices[0].message.content
        
        # 返回 JSON 格式
        return {
            "original_text": input_data.text,
            "sentiment": result,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. 这是一个测试用的首页
@app.get("/")
def read_root():
    return {"message": "我的 AI Agent 正在运行中！"}