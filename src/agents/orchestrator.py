"""
Agent Orchestrator
协调多个agents，管理它们之间的交互和工作流
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .base_agent import BaseAgent, UserContext, AgentResponse
from .body_analysis_agent import BodyAnalysisAgent
from .exercise_plan_agent import ExercisePlanAgent
from .injury_prevention_agent import InjuryPreventionAgent
from .wellness_analysis_agent import WellnessAnalysisAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Agent协调器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化所有agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化所有agents"""
        try:
            # 创建agents
            self.agents['body_analysis'] = BodyAnalysisAgent()
            self.agents['exercise_plan'] = ExercisePlanAgent()
            self.agents['injury_prevention'] = InjuryPreventionAgent()
            self.agents['wellness_analysis'] = WellnessAnalysisAgent()
            
            # 初始化agents
            agent_configs = self.config.get('agents', {})
            for name, agent in self.agents.items():
                agent_config = agent_configs.get(name, {})
                success = agent.initialize(agent_config)
                if not success:
                    self.logger.warning(f"Agent {name} initialization had issues")
            
            self.logger.info(f"Initialized {len(self.agents)} agents")
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise
    
    def process_complete_analysis(self, user_context: UserContext) -> Dict[str, Any]:
        """
        执行完整的分析流程
        
        工作流：
        1. BodyAnalysisAgent - 分析身体情况
        2. ExercisePlanAgent - 基于身体分析生成运动方案
        3. InjuryPreventionAgent - 基于身体分析和运动方案提供预防措施
        4. WellnessAnalysisAgent - 分析运动对身心健康的影响
        
        Args:
            user_context: 用户上下文信息
            
        Returns:
            包含所有agents结果的综合报告
        """
        start_time = datetime.now()
        workflow_id = f"workflow_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting complete analysis workflow: {workflow_id}")
        
        results = {
            'workflow_id': workflow_id,
            'user_id': user_context.user_id,
            'timestamp': start_time.isoformat(),
            'agents': {},
            'summary': {},
            'recommendations': []
        }
        
        try:
            # Step 1: 身体分析
            self.logger.info("Step 1: Running BodyAnalysisAgent...")
            body_analysis_response = self.agents['body_analysis'].process(user_context)
            results['agents']['body_analysis'] = {
                'success': body_analysis_response.success,
                'data': body_analysis_response.data,
                'confidence': body_analysis_response.confidence,
                'reasoning': body_analysis_response.reasoning,
                'recommendations': body_analysis_response.recommendations
            }
            
            if not body_analysis_response.success:
                self.logger.warning("BodyAnalysisAgent failed, continuing with limited data")
            
            body_analysis_data = body_analysis_response.data if body_analysis_response.success else {}
            
            # Step 2: 运动方案
            self.logger.info("Step 2: Running ExercisePlanAgent...")
            exercise_plan_response = self.agents['exercise_plan'].process(
                user_context,
                body_analysis=body_analysis_data
            )
            results['agents']['exercise_plan'] = {
                'success': exercise_plan_response.success,
                'data': exercise_plan_response.data,
                'confidence': exercise_plan_response.confidence,
                'reasoning': exercise_plan_response.reasoning,
                'recommendations': exercise_plan_response.recommendations
            }
            
            exercise_plan_data = exercise_plan_response.data if exercise_plan_response.success else {}
            
            # Step 3: 损伤预防
            self.logger.info("Step 3: Running InjuryPreventionAgent...")
            injury_prevention_response = self.agents['injury_prevention'].process(
                user_context,
                body_analysis=body_analysis_data,
                exercise_plan=exercise_plan_data
            )
            results['agents']['injury_prevention'] = {
                'success': injury_prevention_response.success,
                'data': injury_prevention_response.data,
                'confidence': injury_prevention_response.confidence,
                'reasoning': injury_prevention_response.reasoning,
                'recommendations': injury_prevention_response.recommendations
            }
            
            # Step 4: 身心健康分析
            self.logger.info("Step 4: Running WellnessAnalysisAgent...")
            wellness_response = self.agents['wellness_analysis'].process(
                user_context,
                body_analysis=body_analysis_data,
                exercise_plan=exercise_plan_data
            )
            results['agents']['wellness_analysis'] = {
                'success': wellness_response.success,
                'data': wellness_response.data,
                'confidence': wellness_response.confidence,
                'reasoning': wellness_response.reasoning,
                'recommendations': wellness_response.recommendations
            }
            
            # 生成综合摘要
            results['summary'] = self._generate_summary(results['agents'])
            
            # 整合所有建议
            results['recommendations'] = self._consolidate_recommendations(results['agents'])
            
            # 计算总体置信度
            results['overall_confidence'] = self._calculate_overall_confidence(results['agents'])
            
            # 记录工作流历史
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            workflow_record = {
                'workflow_id': workflow_id,
                'user_id': user_context.user_id,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'processing_time_seconds': processing_time,
                'success': all(
                    agent_result.get('success', False)
                    for agent_result in results['agents'].values()
                )
            }
            self.workflow_history.append(workflow_record)
            
            self.logger.info(f"Workflow {workflow_id} completed in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in workflow {workflow_id}: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results
    
    def _generate_summary(self, agents_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成综合摘要"""
        summary = {
            'body_status': 'unknown',
            'recommended_exercises': [],
            'injury_risk_level': 'unknown',
            'wellness_score': 0.0,
            'key_findings': []
        }
        
        # 身体状态
        if 'body_analysis' in agents_results and agents_results['body_analysis']['success']:
            body_data = agents_results['body_analysis']['data']
            bmi_category = body_data.get('bmi_category', 'unknown')
            summary['body_status'] = bmi_category
            summary['bmi'] = body_data.get('bmi', 0)
            summary['key_findings'].append(f"BMI: {body_data.get('bmi', 0):.1f} ({bmi_category})")
        
        # 推荐运动
        if 'exercise_plan' in agents_results and agents_results['exercise_plan']['success']:
            exercise_data = agents_results['exercise_plan']['data']
            exercises = exercise_data.get('recommended_exercises', [])
            summary['recommended_exercises'] = [
                ex.get('name', 'Unknown') for ex in exercises
            ]
            summary['key_findings'].append(f"推荐{len(exercises)}种运动类型")
        
        # 损伤风险
        if 'injury_prevention' in agents_results and agents_results['injury_prevention']['success']:
            injury_data = agents_results['injury_prevention']['data']
            risk_assessment = injury_data.get('risk_assessment', {})
            summary['injury_risk_level'] = risk_assessment.get('overall_risk_level', 'unknown')
            summary['injury_risk_score'] = risk_assessment.get('risk_score', 0.0)
            summary['key_findings'].append(
                f"损伤风险: {risk_assessment.get('overall_risk_level', 'unknown')}"
            )
        
        # 幸福感评分
        if 'wellness_analysis' in agents_results and agents_results['wellness_analysis']['success']:
            wellness_data = agents_results['wellness_analysis']['data']
            summary['wellness_score'] = wellness_data.get('wellness_score', 0.0)
            overall_wellness = wellness_data.get('overall_wellness', {})
            summary['wellness_level'] = overall_wellness.get('level', 'unknown')
            summary['key_findings'].append(
                f"幸福感评分: {summary['wellness_score']:.2f}/1.0"
            )
        
        return summary
    
    def _consolidate_recommendations(self, agents_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """整合所有agents的建议"""
        all_recommendations = []
        seen_recommendations = set()
        
        for agent_name, agent_result in agents_results.items():
            if agent_result.get('success', False):
                recommendations = agent_result.get('recommendations', [])
                for rec in recommendations:
                    # 去重
                    rec_lower = rec.lower().strip()
                    if rec_lower not in seen_recommendations:
                        all_recommendations.append(rec)
                        seen_recommendations.add(rec_lower)
        
        return all_recommendations
    
    def _calculate_overall_confidence(self, agents_results: Dict[str, Dict[str, Any]]) -> float:
        """计算总体置信度"""
        confidences = []
        for agent_result in agents_results.values():
            if agent_result.get('success', False):
                confidences.append(agent_result.get('confidence', 0.0))
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """获取所有agents的能力描述"""
        capabilities = {}
        for name, agent in self.agents.items():
            capabilities[name] = agent.get_capabilities()
        return capabilities
    
    def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取工作流历史"""
        return self.workflow_history[-limit:]
    
    def process_single_agent(self, agent_name: str, user_context: UserContext,
                           **kwargs) -> AgentResponse:
        """执行单个agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        return agent.process(user_context, **kwargs)

