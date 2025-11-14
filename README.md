# Sports Agent

A multi-agent system for sports health management integrating **specialized agents** and **intelligent orchestration**, providing comprehensive health analysis through body assessment, exercise planning, injury prevention, and wellness analysis.

## Architecture

### Core Components

```
User Input
  ↓
1. BodyAnalysisAgent → Analyze user's physical condition
  ↓
2. ExercisePlanAgent → Recommend optimal exercise plans
  ↓
3. InjuryPreventionAgent → Prevent sports injuries
  ↓
4. WellnessAnalysisAgent → Analyze mental & physical wellness
  ↓
AgentOrchestrator → Coordinate all agents
  ↓
Final Response (Comprehensive Health Analysis)
```



## Key Features Explained

### Multi-Agent Collaboration

Each agent specializes in a specific domain and collaborates seamlessly:

1. **BodyAnalysisAgent** analyzes user's physical condition
2. **ExercisePlanAgent** uses body analysis results to create personalized plans
3. **InjuryPreventionAgent** considers both body analysis and exercise plan to provide prevention measures
4. **WellnessAnalysisAgent** evaluates overall health impact of exercise


## Project Structure

```
Sports_Agent/
├── src/
│   ├── agents/                      # Multi-agent system
│   │   ├── base_agent.py           # Base agent interface
│   │   ├── body_analysis_agent.py   # Body condition analysis
│   │   ├── exercise_plan_agent.py   # Exercise plan recommendation
│   │   ├── injury_prevention_agent.py # Injury prevention
│   │   ├── wellness_analysis_agent.py # Wellness analysis
│   │   └── orchestrator.py         # Agent coordination
│   ├── api/
│   │   └── main.py                 # FastAPI REST API
│   ├── data/                       # Data processing
│   └── core/                       # Core utilities
├── examples/
│   ├── test_multi_agent.py         # System test
│   └── multi_agent_usage.py        # Usage examples
├── configs/                        # Configuration files
├── tests/                          # Unit tests
└── requirements.txt                # Dependencies
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/kevinlmf/Sports_Agent
cd Sports_Agent
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Multi-Agent System

#### Option 1: Test Multi-Agent System (Recommended)

```bash
python examples/test_multi_agent.py
```

This will:
- Initialize all 4 agents
- Run a complete analysis workflow
- Display results from each agent
- Show consolidated recommendations

#### Option 2: Run Usage Examples

```bash
python examples/multi_agent_usage.py
```

This demonstrates:
- Complete analysis workflow
- Single agent usage
- Wellness-focused analysis


## Testing

```bash
# Run all tests
pytest tests/ -v

# Test multi-agent system
python examples/test_multi_agent.py

# Test API
python examples/multi_agent_usage.py
```

## License

This project is licensed under the MIT License.

---
### Disclaimer

This system is for research and educational purposes only. It is not a medical device and must not be used for diagnosis or treatment. Please consult professionals for real medical or mental issues.

---

May we all stay unbroken — in body and in spirit. 💫

