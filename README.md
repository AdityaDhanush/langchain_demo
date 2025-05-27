# LangChain Marketing Analytics Demo

## Overview
This project demonstrates a LangChain-powered analytics assistant for marketing data. It uses a FastAPI backend with a LangChain agent (Google Gemini) and a React frontend.

## Prerequisites
- Docker & Docker Compose
- Google Gemini API Key
- SQLite database file (`ai_ignition.db`)

## Setup
1. Place your `ai_ignition.db` in the `data/` directory.
2. Create a `.env` file:
   ```
   GEMINI_API_KEY=<your_api_key>
   DB_PATH=/app/data/ai_ignition.db
   DEFAULT_BUS_ID=616
   ```
3. Run:
   ```
   docker-compose up --build
   ```
4. Access frontend: http://localhost:3000
   Access API docs: http://localhost:8000/docs

## Example Queries
- "Segment our customers by RFM for Q1 2025."
- "Which products are frequently bought together?"
- "How did each marketing channel perform last week?"
