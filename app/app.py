import uvicorn
import time
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from app.utils.connector import reply_message

from app.utils.rag_utils import get_milvus_retriever, hugging_face_embeddings
from app.utils.chain_utils import get_response_from_llm

load_dotenv()

app = FastAPI()

# API KEYS
LINE_CHATBOT_API_KEY = os.environ.get('LINE_CHATBOT_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPEN_API_KEY = os.environ.get('OPEN_API_KEY')

prompt_template = """
    คุณคือแชทบอทหญิงของบริษัท Chia-Tai โดยมีหน้าที่ให้ความรู้เกี่ยวกับปุ๋ยและโดรนของบริษัท โดยสามารถตอบคำถามจากองค์ความรู้ที่มีอยู่เท่านั้น 
    หากคำถามอยู่นอกเหนือองค์ความรู้ที่มีให้ตอบกลับว่า "ดิฉันไม่สามารถให้คำตอบกับคำถามนี้ได้ ต้องขออภัยเป็นอย่างสูงค่ะ"

    องค์ความรู้ที่เรามีอยู่: 
        {context}

    คำถาม:{input}
    คำตอบ:
"""

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"], max_tokens=512)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
)

class LineEvent(BaseModel):
    replyToken: str
    message: dict
    source: dict
    timestamp: int

class LinePayload(BaseModel):
    events: list[LineEvent]


@app.get("/")
async def hello(): 
    return {"hello": "welcome to LLM chatbot webapp."}

@app.post("/line-webhook")
async def webhook(payload: LinePayload):
    print(payload)
    user_id = payload.events[0].source['userId']
    message_type = payload.events[0].message["type"]
    reply_token = payload.events[0].replyToken

    if message_type == "text":
        """
        If user response as text
        """
        message = payload.events[0].message["text"]
        message_dt = payload.events[0].timestamp
    
        response = get_response_from_llm(llm=llm, input=message)

        print(response)

        await reply_message(user_id, reply_token, reply_message, LINE_CHATBOT_API_KEY)  # Insert Channel access token

        return 200
      
    elif message_type == 'image': 
        """
        If user response as image
        """
        await reply_message(user_id, reply_token, "ขออภัยด้วยค่ะ ฉันไม่สามารถเข้าใจภาพที่คุณส่งมา กรุณาส่งเป็นข้อความแทนนะคะ", LINE_CHATBOT_API_KEY)  # Insert Channel access token

        return 200
    
    elif message_type == 'sticker': 
        """
        If user response as sticker
        """
        await reply_message(user_id, reply_token, "ขออภัยด้วยค่ะ ฉันไม่สามารถเข้าใจ sticker ที่คุณส่งมา กรุณาส่งเป็นข้อความแทนนะคะ", LINE_CHATBOT_API_KEY)  # Insert Channel access token
        
        return 200
    else:
        raise HTTPException(status_code=400, detail="Bad request")
    

if __name__ == "__main__": 
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)