"""
Base Agent Interface for Multi-Agent Sports System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Agent响应数据结构"""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    confidence: float = 0.0  # 0.0-1.0
    reasoning: Optional[str] = None
    recommendations: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    age: int
    gender: str
    height: float  # cm
    weight: float  # kg
    fitness_level: str  # beginner, intermediate, advanced
    health_conditions: List[str] = None
    injury_history: List[Dict[str, Any]] = None
    fitness_goals: List[str] = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.health_conditions is None:
            self.health_conditions = []
        if self.injury_history is None:
            self.injury_history = []
        if self.fitness_goals is None:
            self.fitness_goals = []
        if self.preferences is None:
            self.preferences = {}


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化agent（加载模型、配置等）"""
        pass
    
    @abstractmethod
    def process(self, user_context: UserContext, **kwargs) -> AgentResponse:
        """处理请求并返回结果"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """返回agent的能力描述"""
        pass
    
    def validate_input(self, user_context: UserContext) -> tuple[bool, Optional[str]]:
        """验证输入数据"""
        if not user_context.user_id:
            return False, "user_id is required"
        if user_context.age < 0 or user_context.age > 120:
            return False, "age must be between 0 and 120"
        if user_context.height <= 0 or user_context.height > 300:
            return False, "height must be between 0 and 300 cm"
        if user_context.weight <= 0 or user_context.weight > 300:
            return False, "weight must be between 0 and 300 kg"
        return True, None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

