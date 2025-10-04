import os
import sys
import json
import uvicorn
import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware

# --------------------------- Config ---------------------------
MODEL_REPO = os.getenv("LLAMA_MODEL_REPO", "meta-llama/Meta-Llama-3.3-70B-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 1024))
SUMMARY_END_TOKEN = "<END_OF_SPECS>"

# Load env vars
load_dotenv()

# FastAPI app
app = FastAPI(title="SpecBuddy API", version="1.0")

# Allow frontend (React) to talk to backend (FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Data Classes ---------------------------
@dataclass
class AssistantConfig:
    model_repo: str = MODEL_REPO
    temperature: float = TEMPERATURE
    max_new_tokens: int = MAX_NEW_TOKENS

# --------------------------- Utils ---------------------------
def fetch_image(query: str):
    """Fetch the first image URL for a query using Unsplash API."""
    access_key = os.getenv("UNSPLASH_ACCESS_KEY", "rtmBiR_8-2f0H2MMbJObYI7THw8DUI3Js5mbWF_A3oo")
    if not access_key:
        print("⚠️ Missing Unsplash Access Key. Please set UNSPLASH_ACCESS_KEY in your .env")
        return None

    try:
        url = "https://api.unsplash.com/search/photos"
        params = {"query": query, "per_page": 1}
        headers = {"Authorization": f"Client-ID {access_key}"}

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            return data["results"][0]["urls"]["regular"]
    except Exception as e:
        print("Image search error:", e)

    return None

def make_chat_model(cfg: AssistantConfig):
    """Return a LangChain Groq chat model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Groq API key. Please set GROQ_API_KEY in .env")

    return ChatGroq(
        groq_api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
    )

def build_assistant_chain(chat):
    system_prompt = (
        "You are **SpecBuddy**, a warm, witty, and genuinely curious assistant whose only job is to COLLECT PRODUCT REQUIREMENTS.\n\n"
        "🌟 Personality:\n"
        "- Friendly, conversational, playful tone.\n"
        "- Ask open questions, let user do the talking.\n"
        "- Never recommend or sell products.\n"
        "- Ask delivery mode whether it's pick-up or home delivery.\n"
        "- Always use **bold, large headings** (# Heading) for clarity.\n"
        "- Ask more than two follow up question so the chat feels natural.\n"
        "- Format responses with short sections, bullets, and examples.\n"
        "- Keep answers easy to scan, concise yet engaging.\n\n"
        "📋 Response Style:\n"
        "- Use (#) for large bold headings.\n"
        "- Use (-) or (1.) for bullets.\n"
        "- Give small examples where helpful.\n"
        "- Do not overwhelm the user with too many questions at once.\n\n"
        "👉 When the user types 'done', summarize requirements in **valid minified JSON** and append this token: "
        f"{SUMMARY_END_TOKEN}.\n"
        "{{\"product\": \"string\", \"budget\": \"string\", \"preferred_brands\": [\"string\"], "
        "\"color\": \"string\", \"size\": \"string\", \"Delivery Mode\": \"string\", \"key_specs\": {{\"spec_name\": \"value\"}}}}\n\n"
        "After giving the JSON + <END_OF_SPECS>, continue conversation by asking a friendly **follow-up question specific to that product** "
        "(e.g., if it was a mug: 'Would you like me to suggest matching coasters?' or for an iPhone: 'Do you also need a case for protection?')."
        "⚠️ Never output the words 'undefined', 'null', or similar placeholders."
    )



    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    return prompt | chat | StrOutputParser()

# --------------------------- Globals ---------------------------
cfg = AssistantConfig()
chat = make_chat_model(cfg)
chain = build_assistant_chain(chat)
sessions = {}

def get_session(session_id: str):
    """Return memory + chain for a given session_id, create if not exists"""
    if session_id not in sessions:
        sessions[session_id] = {
            "memory": ConversationBufferMemory(return_messages=True),
            "chain": chain,
        }
    return sessions[session_id]

# --------------------------- API Routes ---------------------------
@app.post("/chat")
def chat_with_assistant(message: str = Body(..., embed=True), session_id: str = Body(..., embed=True)):
    session = get_session(session_id)
    chain = session['chain']
    memory = session['memory']

    # If user finishes
    if message.strip().lower() == "done":
        message = "Please summarize all collected requirements in JSON format now."

    # Save user input
    memory.chat_memory.add_message(HumanMessage(content=message))
    history = memory.load_memory_variables({}).get("history", [])

    # Get assistant reply
    ai_text = chain.invoke({"history": history, "input": message})

    # Save assistant reply
    memory.chat_memory.add_message(AIMessage(content=ai_text))

    # If summary is inside reply
    if SUMMARY_END_TOKEN in ai_text:
        try:
            start = ai_text.find("{")
            end = ai_text.find(SUMMARY_END_TOKEN)
            if start != -1 and end != -1:
                json_blob = ai_text[start:end].strip()
                specs = json.loads(json_blob)

                query = specs.get("product", "")
                if specs.get("color"):
                    query += f" {specs['color']}"
                img_url = fetch_image(query) if query else None

                # ✅ Convert JSON into bulleted summary
                bullet_summary = []
                bullet_summary.append(f"- **Product**: {specs.get('product','N/A')}")
                bullet_summary.append(f"- **Budget**: {specs.get('budget','N/A')}")
                bullet_summary.append(f"- **Preferred Brands**: {', '.join(specs.get('preferred_brands', [])) or 'N/A'}")
                bullet_summary.append(f"- **Color**: {specs.get('color','N/A')}")
                bullet_summary.append(f"- **Size**: {specs.get('size','N/A')}")
                bullet_summary.append(f"- **Delivery Mode**: {specs.get('Delivery Mode','N/A')}")

                if "key_specs" in specs:
                    bullet_summary.append("### Key Specs:")
                    for k, v in specs["key_specs"].items():
                        bullet_summary.append(f"  - {k}: {v}")

                formatted_summary = "\n".join(bullet_summary)

                # ✅ Capture follow-up text safely
                follow_up = ai_text[end + len(SUMMARY_END_TOKEN):].strip()
                if not follow_up or follow_up.lower() in ["undefined", "null", "none"]:
                    follow_up = ""

                # ✅ Build final reply
                combined_reply = f"## 📝 Summary\n{formatted_summary}"
                if follow_up:
                    combined_reply += "\n\n" + follow_up

                return JSONResponse(
                    content={
                        "reply": combined_reply,
                        "summary": specs,
                        "image_url": img_url,
                        "conversation_ended": False,  # keep chat alive
                    }
                )
        except Exception as e:
            return JSONResponse(
                content={"reply": ai_text, "error": f"Failed to parse JSON: {e}"}
            )

    return {"reply": ai_text, "conversation_ended": False}

@app.post("/reset")
def reset_conversation(session_id: str = Body("default", embed=True)):
    """Reset the conversation memory for a given session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": f"Conversation reset for {session_id}"}

if __name__ == "__main__":
    uvicorn.run(app)

