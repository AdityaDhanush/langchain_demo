# app.py

"""
FastAPI service with a single “orchestrator” agent that:
 1. Analyzes the incoming query.
 2. Dispatches it to your four domain-specific agents (segmentation, bundling,
    marketing, data-analysis) as Tools.
 3. Gathers all their outputs and then summarizes them.
"""

import os
import logging
import traceback

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# ——— Import your existing domain-tool lists ———
from project_tools.tools import (
    product_tool,
    segment_tool,
    bundle_tool,
    marketing_tool,
    data_analysis_tool
)

# ——— Logging & FastAPI setup ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orchestrated Multi-Agent Marketing Analytics API",
    description=(
        "A single /query endpoint drives an orchestrator agent which "
        "analyzes your question, fans it out to four sub-agents, then "
        "summarizes all their outputs."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# ——— Global placeholders for agents and chains ———
llm                       = None
segmentation_agent        = None
bundling_agent            = None
marketing_agent           = None
data_agent                = None
summarization_chain       = None
orchestrator_agent        = None


@app.on_event("startup")
def startup_event():
    global llm
    global segmentation_agent, bundling_agent, marketing_agent, data_agent
    global summarization_chain, orchestrator_agent

    # 1) Initialize the shared LLM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY must be set in the environment")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=api_key
    )

    # 2) Build one ZERO_SHOT_REACT agent per domain
    segmentation_agent = initialize_agent(
        segment_tool, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    bundling_agent = initialize_agent(
        bundle_tool, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    marketing_agent = initialize_agent(
        marketing_tool, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    data_agent = initialize_agent(
        data_analysis_tool, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    product_agent = initialize_agent(
        product_tool, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # 3) A simple LLMChain for summarization
    summarize_prompt = PromptTemplate(
        input_variables=["responses"],
        template=(
            "You are an expert summarizer.  "
            "Given the following agent outputs:\n\n{responses}\n\n"
            "Produce a concise but comprehensive summary for the user."
        )
    )
    summarization_chain = LLMChain(
        llm=llm,
        prompt=summarize_prompt,
        verbose=True
    )

    # 4) Wrap each sub-agent as a Tool for the orchestrator
    def call_segmentation(q: str) -> str:
        return segmentation_agent.run(q)

    def call_bundling(q: str) -> str:
        return bundling_agent.run(q)

    def call_marketing(q: str) -> str:
        return marketing_agent.run(q)

    def call_analysis(q: str) -> str:
        return data_agent.run(q)

    def call_product_analysis(q: str) -> str:
        return product_agent.run(q)

    def call_summarize(all_out: str) -> str:
        return summarization_chain.run(responses=all_out)

    orchestrator_tools = [
        Tool(
            name="segmentation",
            func=call_segmentation,
            description="Run the segmentation agent on the user query."
        ),
        Tool(
            name="bundling",
            func=call_bundling,
            description="Run the bundling agent on the user query."
        ),
        Tool(
            name="marketing",
            func=call_marketing,
            description="Run the marketing-analysis agent on the user query."
        ),
        Tool(
            name="analysis",
            func=call_analysis,
            description="Run the data-analysis agent on the user query."
        ),
        Tool(
            name="product specific analysis",
            func=call_product_analysis,
            description="Run the product specific analysis agent on the user query."
        ),
        Tool(
            name="summarize",
            func=call_summarize,
            description="Summarize all prior agent outputs for the user."
        ),
    ]

    # 5) Initialize the orchestrator as a ZERO-SHOT agent
    orchestrator_agent = initialize_agent(
        orchestrator_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )


@app.post("/query")
async def query(request: QueryRequest):
    """
    The orchestrator agent will:
      1) Think through (chain-of-thought) how to break the query
         into sub-tasks (actions).
      2) Invoke each Tool (i.e. sub-agent) in turn.
      3) Finally call “summarize” on the gathered outputs.
    """
    try:
        raw = orchestrator_agent.run(request.query)
        # If LangChain returns ToolResponse, normalize:
        result = raw.output if hasattr(raw, "output") else str(raw)
        return {"result": result}

    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
