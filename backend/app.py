import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# TODO: Replace ChatOpenAI with actual Google Gemini integration
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from tools import segment_customers, find_bundles, analyze_marketing

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY is not set")
    global agent
    # Using ChatOpenAI as placeholder; replace with Google Gemini model in production
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    tools = [
        Tool(name="segment_customers", func=segment_customers,
             description="Segment customers by RFM metrics."),
        Tool(name="find_bundles", func=find_bundles,
             description="Identify product bundles via association rules."),
        Tool(name="analyze_marketing", func=analyze_marketing,
             description="Analyze marketing channels and conversion metrics."),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = agent.run(request.query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
