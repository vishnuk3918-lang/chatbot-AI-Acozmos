import os
import json
import requests
import uvicorn
from dataclasses import dataclass
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# --------------------------- Config ---------------------------
load_dotenv()

MODEL_REPO = os.getenv("LLAMA_MODEL_REPO", "meta-llama/Meta-Llama-3.3-70B-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 1024))
SUMMARY_END_TOKEN = "<END_OF_SPECS>"

# --------------------------- FastAPI Setup ---------------------------
app = FastAPI(title="SpecBuddy API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Data Class ---------------------------
@dataclass
class AssistantConfig:
    model_repo: str = MODEL_REPO
    temperature: float = TEMPERATURE
    max_new_tokens: int = MAX_NEW_TOKENS

# --------------------------- Utils ---------------------------
def make_chat_model(cfg: AssistantConfig):
    """Initialize the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Missing GROQ_API_KEY in .env file")

    return ChatGroq(
        groq_api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
    )

def fetch_image(query: str):
    """Fetch image from Unsplash based on LLM-generated prompt."""
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        print("⚠️ Missing Unsplash Access Key")
        return None

    try:
        url = "https://api.unsplash.com/search/photos"
        params = {"query": query, "per_page": 1}
        headers = {"Authorization": f"Client-ID {access_key}"}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            return data["results"][0]["urls"]["regular"]
    except Exception as e:
        print("Image fetch failed:", e)
    return None

# --------------------------- LLM Helpers ---------------------------
def generate_summary_with_llm(chat, history):
    """Use SpecBuddy rules to generate the final summary."""
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are **SpecBuddy**, a warm, witty, and genuinely curious assistant that collects and summarizes product requirements.\n\n"
            "🌟 Personality:\n"
            "- Friendly, conversational, and professional tone.\n"
            "- You are talking to Indian users — always think and speak in Indian context.\n"
            "- All prices, budgets, and amounts must always be expressed in **Indian Rupees (₹)**.\n"
            "- ⚠️ Never mention or ask about dollars ($), USD, or foreign currencies. If the user mentions dollars, automatically convert them (1 USD ≈ ₹83).\n\n"
            "💬 Conversation Flow:\n"
            "- Ask short, open-ended questions to understand the user’s product needs.\n"
            "- Ask only two or three folow up questions so it feels like a natural conversation.\n"
            "- After asking follow-up questions, ask do you want to add anything else?\n"
            "- When asking about delivery, only offer two valid options: **Home Delivery** 🏠 or **Pickup from Store** 🏬.\n"
            "- ⚠️ Never mention or suggest online marketplaces like Amazon, Flipkart, or e-commerce websites.\n"
            "- Do not recommend where to buy or sell — your role is only to collect product requirements.\n"
            "- After gathering enough details, ask if the user wants to add anything else before summarizing.\n"
            "- When the user types 'done', summarize their requirements clearly.\n\n"
            "📋 Summary Rules:\n"
            "- If the user is a buyer, start the summary with '🧾 Buyer Summary'.\n"
            "- If the user is a seller, start the summary with ' 🧾 Seller Summary'.\n"
            "- Always display budgets and prices in Indian Rupees (₹).\n"
            "- Present details in a neat Markdown list using bullet points.\n"
            "- After the summary, append the token <END_OF_SPECS>.\n"
            "- Then, write one friendly follow-up question separately, prefixed with <FOLLOW_UP>.\n\n"
            "⚠️ Do not output JSON. The entire summary and follow-up must be in plain Markdown text.\n"
            "💡 Example: Instead of saying 'Do you want to buy from Amazon?', say 'Would you prefer home delivery 🏠 or pickup from store 🏬?'"
         ),
        ("human", "{history}")
    ])
    chain = summary_prompt | chat | StrOutputParser()
    summary = chain.invoke({"history": str(history)})
    return summary.strip()

def generate_image_query_with_llm(chat, summary_text):
    """Ask LLM to create a descriptive image query."""
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the product summary, produce a concise descriptive phrase for Unsplash image search."),
        ("human", f"Summary:\n{summary_text}\n\nReturn only the search query, e.g., 'sleek silver gaming laptop'.")
    ])
    chain = query_prompt | chat | StrOutputParser()
    result = chain.invoke({})
    return result.strip().strip('"')

# --------------------------- Globals ---------------------------
cfg = AssistantConfig()
chat = make_chat_model(cfg)
sessions = {}

def get_session(session_id: str):
    """Retrieve or create chat session memory."""
    if session_id not in sessions:
        sessions[session_id] = {"memory": ConversationBufferMemory(return_messages=True)}
    return sessions[session_id]

# --------------------------- Routes ---------------------------
@app.post("/chat")
def chat_with_assistant(message: str = Body(..., embed=True), session_id: str = Body(..., embed=True)):
    """Main chat endpoint for SpecBuddy."""
    session = get_session(session_id)
    memory = session["memory"]

    # Record user message
    memory.chat_memory.add_message(HumanMessage(content=message))
    history = memory.load_memory_variables({}).get("history", [])

    # If user finishes
    if message.strip().lower() == "done":
        summary = generate_summary_with_llm(chat, history)
        image_query = generate_image_query_with_llm(chat, summary)
        image_url = fetch_image(image_query)
        return JSONResponse(content={
            "reply": summary,
            "image_query": image_query,
            "image_url": image_url
        })

    # Normal conversation with SpecBuddy style
    convo_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are **SpecBuddy**, a warm, witty, and genuinely curious assistant that collects and summarizes product requirements.\n"
         "Ask friendly, short questions to learn about what the user wants to buy or sell.\n"
         "Encourage them to specify product, brand, budget, color, and delivery mode.\n"
         "Do not summarize yet until the user says 'done'."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = convo_prompt | chat | StrOutputParser()
    reply = chain.invoke({"history": history, "input": message})

    # Save assistant reply
    memory.chat_memory.add_message(AIMessage(content=reply))

    return {"reply": reply}

@app.post("/sales_trainer")
def sales_trainer(message: str = Body(..., embed=True), session_id: str = Body(..., embed=True)):
    """Interactive sales training mode — teaches persuasive selling techniques."""

    session = get_session(session_id)
    memory = session["memory"]

    # Record the student's message
    memory.chat_memory.add_message(HumanMessage(content=message))
    history = memory.load_memory_variables({}).get("history", [])

    # If user finishes the session
    if message.strip().lower() == "done":
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a legendary **Sales Teacher**, a master mentor who trains sellers to use psychology, emotion, and storytelling to sell effectively.\n\n"
             "🧠 **Style & Role:**\n"
             "- You are the **Teacher**, the user is your **Student Seller**.\n"
             "- Speak like a warm, wise mentor.\n"
             "- Be conversational, use examples, and challenge the student with real-life sales situations.\n"
             "- Reinforce key ideas: never sell price, always sell **value**, **results**, and **emotion**.\n"
             "- Use Indian context (rupees ₹, weddings, local situations, etc.).\n\n"
             "💬 Example Dialogue (for reference, this guides your tone):\n"
             "Teacher: What do you do, ma'am?\n"
             "Student: I'm a makeup artist.\n"
             "Teacher: You're a makeup artist? Fantastic. How much do you charge per hour?\n"
             "Student: ₹30,000.\n"
             "Teacher: And how many customers have asked you to lower your price?\n"
             "Student: Quite a few.\n"
             "Teacher: Never say no. Instead, make them see the value. For example — 'You're going to a wedding? Is your ex going to be there? How much is it worth for him to see you looking your best?' You’ve shifted the talk from **price to value**.\n\n"
             "🪄 **Summary Task:**\n"
             "When the student types 'done', summarize what they learned from this session in an inspiring tone.\n"
             "End with one motivational line, like: 'Remember, people don’t buy products — they buy feelings.'"
             ),
            ("human", str(history))
        ])
        chain = summary_prompt | chat | StrOutputParser()
        reply = chain.invoke({})
        return {"reply": reply.strip()}

    # Normal interactive conversation
    convo_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are **The Great Sales Teacher**, coaching your student on persuasive selling using the power of storytelling.\n\n"
         "🎯 **Your Teaching Style:**\n"
         "- Speak as 'Teacher', reply to the user's messages as your student.\n"
         "- Teach through dialogue, like this example:\n\n"
         "Teacher: What do you do, ma'am?\n"
         "Student: I'm a makeup artist.\n"
         "Teacher: Great! So, if someone says, 'Your price is too high,' do you argue or reframe it?\n"
         "Student: I say I can’t lower it.\n"
         "Teacher: No, you don’t say no — you change the conversation from **price to value**. Ask them emotional questions that make them rethink, e.g., 'Who’s the makeup for? Your wedding? Is your ex coming?' Suddenly, the price feels worth it.\n\n"
         "✨ **Your Goal:**\n"
         "- Teach the student to sell outcomes, not features.\n"
         "- Reinforce that customers buy transformation, emotion, and value — not price.\n"
         "- Use Indian examples and Rupee (₹) pricing.\n"
         "- Stay warm, witty, and motivational.\n"
         "- End each reply naturally, as if in an ongoing teacher–student conversation.\n"
         "- Never summarize until the student says 'done'."
         ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = convo_prompt | chat | StrOutputParser()
    reply = chain.invoke({"history": history, "input": message})

    # Save teacher's reply
    memory.chat_memory.add_message(AIMessage(content=reply))

    return {"reply": reply}


@app.post("/reset")
def reset_conversation(session_id: str = Body("default", embed=True)):
    """Reset conversation memory."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": f"Conversation reset for {session_id}"}

# --------------------------- Run ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
