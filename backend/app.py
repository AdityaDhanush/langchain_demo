import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# TODO: Replace ChatOpenAI with actual Google Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from tools import segment_customers, find_bundles, analyze_marketing

import logging, traceback
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-pro",  # or "gemini-pro"
        google_api_key = api_key
    )
    tools = [
        Tool(name="segment_customers", func=segment_customers,
             description="Segment customers by RFM metrics."),
        Tool(name="find_bundles", func=find_bundles,
             description="Identify product bundles via association rules."),
        Tool(name="analyze_marketing", func=analyze_marketing,
             description="Analyze marketing channels and conversion metrics."),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# @app.post("/query")
# async def query(request: QueryRequest):
#     try:
#         result = agent.run(request.query)
#         return {"result": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    try:
        # If you want to silence the deprecation warning, you could later switch to agent.invoke()
        agent_response = agent.invoke(request.query)
        # result = agent_response.output if hasattr(agent_response, "output") else agent_response

        # If it has an `.output` attribute, use that; otherwise stringify
        if hasattr(agent_response, "output"):
            final = agent_response.output
        elif isinstance(agent_response, dict) and "output" in agent_response:
            final = agent_response["output"]
        else:
            final = str(agent_response)
        return {"result": final}

    except Exception as e:
        # Log full traceback to container logs
        tb = traceback.format_exc()
        logger.error(f"Error while running agent:\n{tb}")
        # Return the error message (and optionally traceback) in the HTTP response
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                # "traceback": tb  # you can include this if you want it on the client
            }
        )

