"""
Runtime Service — FastAPI entry point.

This is the ONLY file that starts the server.
All agent logic is delegated to RuntimeEngine via web/routes.py.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.routes import router

app = FastAPI(
    title="Agent Runtime Service",
    description="REST API for the Agent Runtime Kernel (Causal Chain)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow cross-origin requests (for local dev / future gateway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all Runtime API routes
app.include_router(router)


@app.get("/health")
async def root_health():
    return {"status": "ok", "service": "agent-runtime"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
