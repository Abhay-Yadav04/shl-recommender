from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, Recommendation
from agent import get_agent_reply

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Validate roles
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'.",
            )

    # Last message must be from user
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user.")

    result = get_agent_reply(request.messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )
