#!/usr/bin/env python
"""
API Demo for Multi-Agent Sports Health Management System
Demonstrates API endpoints for multi-agent system
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from typing import Dict, List, Optional

def demo_api_structure():
    """Demonstrate API endpoint structure"""
    print("\n" + "="*70)
    print("Multi-Agent System API Architecture")
    print("="*70 + "\n")

    api_structure = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Multi-Agent Sports Health Management API",
    description="Multi-agent system for sports health management",
    version="2.0.0"
)

class UserInput(BaseModel):
    user_id: str
    age: int
    gender: str
    height: float
    weight: float
    fitness_level: str
    health_conditions: List[str] = []
    injury_history: List[Dict] = []
    fitness_goals: List[str] = []
    preferences: Dict = {}

@app.post("/api/v2/analyze")
async def complete_analysis(user_input: UserInput):
    # Run all agents
    # Return comprehensive analysis
    pass

@app.post("/api/v2/agents/{agent_name}")
async def single_agent_analysis(agent_name: str, user_input: UserInput):
    # Run single agent
    pass

@app.get("/api/v2/agents")
async def list_agents():
    # List all available agents
    pass

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
"""

    print("API Endpoints:")
    print()
    print("POST /api/v2/analyze")
    print("  • Complete multi-agent analysis")
    print("  • Returns comprehensive health assessment")
    print()
    print("POST /api/v2/agents/{agent_name}")
    print("  • Single agent analysis")
    print("  • Available agents: body_analysis, exercise_plan, injury_prevention, wellness_analysis")
    print()
    print("GET /api/v2/agents")
    print("  • List all available agents and their capabilities")
    print()
    print("GET /health")
    print("  • Health check endpoint")

    print("\n✓ API structure defined")
    print("  • Framework: FastAPI")
    print("  • Validation: Pydantic models")
    print("  • Documentation: Auto-generated (Swagger UI)")


def demo_request_examples():
    """Demonstrate API request/response examples"""
    print("\n" + "="*70)
    print("API Request/Response Examples")
    print("="*70 + "\n")

    # Example 1: Complete analysis
    request_complete = {
        "user_id": "user_001",
        "age": 30,
        "gender": "male",
        "height": 175.0,
        "weight": 75.0,
        "fitness_level": "intermediate",
        "health_conditions": [],
        "injury_history": [],
        "fitness_goals": ["weight_loss", "muscle_gain"],
        "preferences": {}
    }

    response_complete = {
        "workflow_id": "workflow_20250117_120000",
        "user_id": "user_001",
        "overall_confidence": 0.92,
        "summary": {
            "body_status": "normal",
            "bmi": 24.5,
            "recommended_exercises": ["有氧运动", "力量训练"],
            "injury_risk_level": "low",
            "wellness_score": 0.67
        },
        "recommendations": [
            "建议每周进行4次训练，每次45分钟",
            "训练前务必进行5-10分钟热身",
            "规律运动对身心健康有显著益处"
        ]
    }

    print("Example 1: Complete Multi-Agent Analysis")
    print(f"Request: {json.dumps(request_complete, indent=2)}")
    print(f"\nResponse: {json.dumps(response_complete, indent=2)}")

    # Example 2: Single agent
    request_single = {
        "user_id": "user_002",
        "age": 25,
        "gender": "female",
        "height": 165.0,
        "weight": 60.0,
        "fitness_level": "beginner",
        "health_conditions": [],
        "injury_history": [],
        "fitness_goals": ["general_fitness"],
        "preferences": {}
    }

    response_single = {
        "agent_name": "body_analysis",
        "success": True,
        "data": {
            "bmi": 22.0,
            "bmi_category": "normal",
            "health_risks": {
                "overall_risk_level": "low",
                "risk_score": 0.2
            }
        },
        "confidence": 0.85,
        "recommendations": [
            "保持当前的健康生活方式，定期运动"
        ]
    }

    print("\n\nExample 2: Single Agent Analysis (Body Analysis)")
    print(f"Request: POST /api/v2/agents/body_analysis")
    print(f"Body: {json.dumps(request_single, indent=2)}")
    print(f"\nResponse: {json.dumps(response_single, indent=2)}")

    print("\n✓ Request/response examples provided")


def main():
    """Run API demo"""
    print("\n" + "="*70)
    print("Multi-Agent Sports Health Management - API Demo")
    print("="*70)

    demo_api_structure()
    demo_request_examples()

    print("\n" + "="*70)
    print("API Demo Complete")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ FastAPI service for multi-agent system")
    print("  ✓ RESTful API endpoints")
    print("  ✓ Comprehensive health analysis")
    print("\nNext Steps:")
    print("  • Start API server: uvicorn src.api.main:app --reload")
    print("  • Access API docs: http://localhost:8000/docs")
    print("  • Test endpoints with examples above")
    print()


if __name__ == "__main__":
    main()
