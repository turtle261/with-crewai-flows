import os
import uvicorn
from ag_ui_crewai.endpoint import add_crewai_flow_fastapi_endpoint
from fastapi import FastAPI
from dotenv import load_dotenv
from agent import SampleAgentFlow

load_dotenv()

app = FastAPI()
add_crewai_flow_fastapi_endpoint(app, SampleAgentFlow(), "/")

def main():
  """Run the uvicorn server."""
  port = int(os.getenv("PORT", "8007"))
  uvicorn.run(
    "server:app",
    host="0.0.0.0",
    port=port,
    reload=True,
  )

if __name__ == "__main__":
  main()