from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from groq import Groq

app = FastAPI()
client = Groq(api_key="API KEY HERE") 

# Paste the FULL behavior string here
SYSTEM_PROMPT = (
    "You are an intelligent AI heritage guide for the 'LUXORA' mobile application. "
    "Your goal is to assist tourists and history enthusiasts exploring Egyptian monuments, "
    "specifically in Luxor. Provide accurate, engaging, and verified historical context "
    "about temples, tombs, and hieroglyphic inscriptions. "
    "CRITICAL RULES: "
    "1. Always respond in the SAME LANGUAGE the user uses to ask the question. "
    "2. Keep your answers concise, friendly, and easy to understand for tourists. "
    "3. STRICTLY OUT OF SCOPE: If the user asks about ANY topic unrelated to Egyptology, Luxor, tourism, or ancient monuments (e.g., coding, sports, cooking, general weather in other countries), you MUST politely REFUSE to answer. State clearly that you are an AI guide dedicated only to Luxor's heritage, and invite them to ask a historical question instead."
)

# Define the structure for a single message
class Message(BaseModel):
    role: str
    content: str

# Update the request to accept a list of messages
class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/")
async def root():
    return {"status": "LUXORA Bot is Online"}

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        # Start with the system prompt
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Append the chat history sent from the mobile app
        for msg in request.messages:
            conversation.append({"role": msg.role, "content": msg.content})

        completion = client.chat.completions.create(
            messages=conversation,
            model="llama-3.3-70b-versatile",
            temperature=0.7 # Added the temperature from your notebook script for consistency
        )
        return {"bot_message": {"message": completion.choices[0].message.content}}
    
    except Exception as e:
        print(f"Groq Error: {e}") 
        raise HTTPException(status_code=500, detail=str(e))