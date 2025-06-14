from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock employee database
employees = [
    {
        "id": 1,
        "name": "John Doe",
        "role": "Software Engineer",
        "department": "Engineering",
        "experience": "5 years",
        "skills": ["Python", "React", "AWS"],
        "location": "New York",
        "team_size": 6
    },
    {
        "id": 2,
        "name": "Jane Smith",
        "role": "Product Manager",
        "department": "Product",
        "experience": "7 years",
        "skills": ["Product Strategy", "Agile", "User Research"],
        "location": "San Francisco",
        "team_size": 8
    },
    # Add more employees as needed
]

class ChatRequest(BaseModel):
    message: str
    top_k: int = 5

class ChatResponse(BaseModel):
    response: str
    thoughts: List[str]
    results: Optional[List[dict]] = None

def search_employees(query: dict) -> List[dict]:
    """
    Search employees based on query parameters
    """
    results = employees.copy()
    
    if query.get("role"):
        results = [e for e in results if query["role"].lower() in e["role"].lower()]
    
    if query.get("department"):
        results = [e for e in results if query["department"].lower() in e["department"].lower()]
    
    if query.get("experience_min"):
        results = [e for e in results if int(e["experience"].split()[0]) >= query["experience_min"]]
    
    if query.get("skills"):
        results = [e for e in results if any(skill.lower() in [s.lower() for s in e["skills"]] for skill in query["skills"])]
    
    if query.get("location"):
        results = [e for e in results if query["location"].lower() in e["location"].lower()]
    
    return results

def analyze_message(message: str) -> dict:
    """
    Analyze user message to determine intent and extract search parameters
    """
    # In a real application, you would use NLP here
    # This is a simple keyword-based analysis
    query = {}
    
    message = message.lower()
    
    if "engineer" in message:
        query["role"] = "engineer"
    if "manager" in message:
        query["role"] = "manager"
    if "product" in message:
        query["department"] = "product"
    if "engineering" in message:
        query["department"] = "engineering"
    
    # Extract experience requirement
    if "years" in message:
        words = message.split()
        for i, word in enumerate(words):
            if word == "years" and i > 0:
                try:
                    query["experience_min"] = int(words[i-1])
                except ValueError:
                    pass
    
    return query

def generate_followup_question(current_query: dict) -> Optional[str]:
    """
    Generate follow-up questions based on missing information
    """
    if not current_query:
        return "What type of role are you looking for?"
    
    if "role" in current_query and "department" not in current_query:
        return f"Which department should the {current_query['role']} be in?"
    
    if "role" in current_query and "experience_min" not in current_query:
        return f"How many years of experience should the {current_query['role']} have?"
    
    if "role" in current_query and "skills" not in current_query:
        return f"What skills are you looking for in the {current_query['role']}?"
    
    return None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    thoughts = ["Analyzing user message..."]
    time.sleep(0.5)  # Simulate processing time
    
    # Analyze the message
    query = analyze_message(request.message)
    thoughts.append(f"Extracted search parameters: {query}")
    time.sleep(0.5)
    
    # Generate follow-up question or search
    followup = generate_followup_question(query)
    if followup:
        thoughts.append("Generating follow-up question...")
        return ChatResponse(
            response=followup,
            thoughts=thoughts,
            results=[]
        )
    
    # Perform search
    thoughts.append("Searching employee database...")
    results = search_employees(query)
    thoughts.append(f"Found {len(results)} matching employees")
    
    response = (
        f"I found {len(results)} employees matching your criteria. "
        "You can see their details below. Would you like to refine your search further?"
    )
    
    return ChatResponse(
        response=response,
        thoughts=thoughts,
        results=results[:request.top_k]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)