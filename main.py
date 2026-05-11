from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>SHL Assessment Recommender</title>
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f7fb; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .card { background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 48px 40px; max-width: 560px; width: 100%; }
        .logo { font-size: 28px; font-weight: 800; color: #1F3864; margin-bottom: 6px; }
        .logo span { color: #2E75B6; }
        .subtitle { color: #555; font-size: 15px; margin-bottom: 32px; }
        h2 { font-size: 14px; font-weight: 700; color: #1F3864; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 12px; }
        .endpoint { background: #f4f7fb; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; }
        .method { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; margin-right: 8px; }
        .get { background: #d1fae5; color: #065f46; }
        .post { background: #dbeafe; color: #1e40af; }
        .path { font-family: monospace; font-size: 14px; color: #1F3864; }
        .desc { font-size: 13px; color: #666; margin-top: 5px; }
        .status { display: flex; align-items: center; gap: 8px; margin-top: 28px; padding-top: 20px; border-top: 1px solid #eee; font-size: 13px; color: #555; }
        .dot { width: 9px; height: 9px; border-radius: 50%; background: #22c55e; }
      </style>
    </head>
    <body>
      <div class="card">
        <div class="logo">SHL<span>.</span> Assessment Recommender</div>
        <div class="subtitle">Conversational agent for SHL assessment selection — AI Intern Assignment</div>

        <h2>API Endpoints</h2>

        <div class="endpoint">
          <span class="method get">GET</span><span class="path">/health</span>
          <div class="desc">Readiness check — returns <code>{"status": "ok"}</code></div>
        </div>

        <div class="endpoint">
          <span class="method post">POST</span><span class="path">/chat</span>
          <div class="desc">Accepts full conversation history, returns agent reply and structured assessment recommendations.</div>
        </div>

        <div class="endpoint">
          <span class="method get">GET</span><span class="path">/docs</span>
          <div class="desc">Interactive API documentation (Swagger UI)</div>
        </div>

        <div class="status">
          <div class="dot"></div>
          Service is live and accepting requests
        </div>
      </div>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'.",
            )

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user.")

    result = get_agent_reply(request.messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )
