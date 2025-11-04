from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os
from typing import Optional, List
from pydantic import BaseModel
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

# -------------------------------
# ENV Config
# -------------------------------
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
CHATWOOT_API_ACCESS_TOKEN = os.getenv("CHATWOOT_API_ACCESS_TOKEN")
CHATWOOT_INBOX_ID = os.getenv("CHATWOOT_INBOX_ID")
CHATWOOT_BASE_URL = f"http://16.170.159.71:3000/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}"
CHATWOOT_HEADERS = {
    "Content-Type": "application/json",
    "api_access_token": CHATWOOT_API_ACCESS_TOKEN,
}

class SentenceTransformerEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

# -------------------------------
# Chroma Initialization
# -------------------------------
embedding_function = SentenceTransformerEmbeddings()
chroma_client = Client(
    Settings(
        is_persistent=True,
        persist_directory="chroma_db"
    )
)


# -------------------------------
# Mistral AI Configuration
# -------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

async def generate_mistral_response(context: str, user_message: str):
    prompt = f"""
    You are Raptee.HV's intelligent assistant.
    Use the following context to answer precisely.
    Be professional and give response only from the given context, also speak only about raptee.
    If you don't know the answer, politely say so and suggest contacting support.

    Context:
    {context}

    User Question:
    {user_message}
    """

    try:
        response = mistral_client.chat.complete(
            model="mistral-medium-latest", 
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        
        print("DEBUG: Mistral chat returned success but empty content.")
        return "I apologize, but I'm having trouble generating a response from the model. Please try again later or contact support."
        
    except Exception as e:
        print(f"Mistral generation error (Chat Endpoint): {e}")
        return "I apologize, but I'm having trouble processing your request. Please try again later or contact support."
    

# -------------------------------
# Chatwoot Helpers
# -------------------------------
async def get_conversation(conversation_id: str) -> Optional[dict]:
    url = f"{CHATWOOT_BASE_URL}/conversations/{conversation_id}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=CHATWOOT_HEADERS)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error fetching conversation: {e}")
        return None

async def send_chatwoot_message(conversation_id: str, text: str) -> None:
    url = f"{CHATWOOT_BASE_URL}/conversations/{conversation_id}/messages"
    payload = {"content": text, "message_type": "outgoing"}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload, headers=CHATWOOT_HEADERS)
    except Exception as e:
        print(f"Error sending message: {e}")

async def toggle_conversation_status(conversation_id: str, status: str) -> None:
    url = f"{CHATWOOT_BASE_URL}/conversations/{conversation_id}/toggle_status"
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json={"status": status}, headers=CHATWOOT_HEADERS)
        print(f"Conversation {conversation_id} status set to: {status}")
    except Exception as e:
        print(f"Error toggling status: {e}")

async def handoff_to_human(conversation_id: str) -> None:
    await send_chatwoot_message(
        conversation_id,
        "I'm connecting you to a human agent now. Please wait a moment! 👤"
    )
    await toggle_conversation_status(conversation_id, "open")

# -------------------------------
# RAG Retrieval
# -------------------------------
async def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[str]:
    try:
        collection = chroma_client.get_collection(
            name="raptee_t30_docs",
            embedding_function=embedding_function
        )
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"Chroma retrieval failed: {e}")
        return []

# -------------------------------
# FastAPI Webhook Handler
# -------------------------------
app = FastAPI()

class WebhookPayload(BaseModel):
    event: str
    message_type: str
    conversation: Optional[dict] = None
    content: Optional[str] = None

@app.get("/test")
async def test_collection():
    try:
        chunks = await retrieve_relevant_chunks("What is the T30?")
        return {"chunks": chunks}
    except Exception as e:
        return {"error": str(e)}

@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    if payload.get("event") != "message_created" or payload.get("message_type") != "incoming":
        return JSONResponse(content={"status": "Ignored non-incoming message"}, status_code=200)

    conversation_id = payload.get("conversation", {}).get("id")
    user_message = payload.get("content")

    if not conversation_id or not user_message:
        raise HTTPException(status_code=400, detail="Invalid payload")

    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if any(keyword in user_message.lower() for keyword in ["human", "agent", "support", "help", "person"]):
        await handoff_to_human(conversation_id)
        return JSONResponse(content={"status": "Handoff initiated"}, status_code=200)

    top_chunks = await retrieve_relevant_chunks(user_message, 5)
    context = "\n\n---\n\n".join(top_chunks)
    bot_response = await generate_mistral_response(context, user_message)
    await send_chatwoot_message(conversation_id, bot_response)

    return JSONResponse(content={"status": "Message processed"}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
