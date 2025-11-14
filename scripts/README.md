# Multi-Agent Sports Health Management - Demo Scripts

This directory contains demonstration scripts for the Multi-Agent Sports Health Management System.

## 📋 Overview

The system features:

1. **Multi-Agent System** - Four specialized agents working together
2. **API/Deployment** - Production-ready FastAPI service
3. **Agent Orchestration** - Workflow management and coordination

## 🚀 Quick Start

### Run Complete Demo

```bash
# Run multi-agent system demo
./scripts/run_complete_demo.sh
```

### Run Individual Demos

```bash
# API: Deployment architecture
python scripts/demo_api.py

# Enterprise: Interpretability and validation
python scripts/demo_enterprise.py

# Test multi-agent system
python examples/test_multi_agent.py
```

## 📁 Script Descriptions

### `run_complete_demo.sh`
Master script that demonstrates the multi-agent system capabilities.

**Features:**
- Color-coded output for easy reading
- Progressive execution with status updates
- Checks system capabilities
- Generates comprehensive summary report

**Output:**
- System status summary table
- Agent capabilities overview
- Test coverage report

### `demo_api.py`
Showcases production deployment architecture.

**Components:**
- **FastAPI Service**: REST API endpoints for multi-agent system
- **Request/Response Examples**: User input and agent responses
- **API Documentation**: Auto-generated Swagger docs

**Usage:**
```python
python scripts/demo_api.py
```

### `test_multi_agent.py`
Tests the multi-agent system functionality.

**Components:**
- Complete analysis workflow
- Individual agent testing
- Response validation

**Usage:**
```python
python examples/test_multi_agent.py
```

## 🎯 What Each Demo Shows

### Multi-Agent System

**Agents:**
- ✅ BodyAnalysisAgent - Body condition analysis
- ✅ ExercisePlanAgent - Exercise plan recommendation
- ✅ InjuryPreventionAgent - Injury risk prevention
- ✅ WellnessAnalysisAgent - Mental and physical wellness analysis

**Orchestration:**
- Workflow management
- Agent coordination
- Result aggregation

### API/Deployment

**API Endpoints:**
```
POST /api/v2/analyze          - Complete multi-agent analysis
POST /api/v2/agents/{name}    - Single agent analysis
GET  /api/v2/agents           - List all agents
GET  /api/v2/workflow/history - Workflow history
GET  /health                  - Health check
```

## 📊 Expected Output

### System Status Summary
```
┌─────────────────────────────────────────────────────────────────────┐
│                      SYSTEM STATUS SUMMARY                          │
├─────────────────────────────────────────────────────────────────────┤
│ Component                │ Status │ Key Feature                     │
├──────────────────────────┼────────┼─────────────────────────────────┤
│ Multi-Agent System       │   ✅   │ 4 Specialized Agents            │
│ API / Deployment         │   ✅   │ FastAPI REST API                 │
│ Agent Orchestration      │   ✅   │ Workflow Management              │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Prerequisites

### Required Dependencies
```bash
# Core
pip install fastapi uvicorn pydantic

# Data processing
pip install pandas numpy
```

### System Requirements
- Python 3.8+
- 4GB+ RAM

## 📝 Usage Examples

### Basic Demo Run
```bash
# Quick test - multi-agent system
python examples/test_multi_agent.py

# API demo
python scripts/demo_api.py

# Full system demo
./scripts/run_complete_demo.sh
```

### Start API Server
```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --port 8000

# Access API docs
# http://localhost:8000/docs
```

## 🐛 Troubleshooting

### Script Permission Error
```bash
chmod +x scripts/run_complete_demo.sh
```

### Missing Dependencies
```bash
pip install fastapi uvicorn pydantic pandas numpy
```

### Import Errors
- Ensure you're running from the project root directory
- Check that `src/` is in your Python path

## 📚 Related Documentation

- **README.md** - Main project overview
- **examples/test_multi_agent.py** - Multi-agent system test
- **examples/multi_agent_usage.py** - Usage examples

## 🎓 Learning Path

**Beginners:**
1. Run `python examples/test_multi_agent.py` to see agents in action
2. Study `src/agents/` for agent implementation
3. Explore API endpoints at `http://localhost:8000/docs`

**Intermediate:**
1. Review `src/agents/orchestrator.py` for workflow management
2. Study individual agent implementations
3. Customize agent behavior via configuration

**Advanced:**
1. Create custom agents by extending `BaseAgent`
2. Implement new agent capabilities
3. Extend orchestrator with custom workflows

## 🤝 Contributing

If you add new features, please update the corresponding demo script:
- New agents → Update agent tests
- New API endpoints → Update `demo_api.py`
- New workflows → Update orchestrator tests

## 📧 Support

For issues or questions:
1. Check this README and main README.md
2. Review example scripts for usage patterns
3. Inspect source code in `src/agents/`

---

**Last Updated:** 2025-01-17
**Status:** ✅ Multi-agent system functional
**Version:** 2.0.0
