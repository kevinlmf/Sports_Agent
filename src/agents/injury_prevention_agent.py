"""
Injury Prevention Agent
负责防止运动损伤，包括：
- 识别运动损伤风险因素
- 提供预防措施建议
- 监控训练负荷
- 推荐恢复策略
"""

from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResponse, UserContext
import logging

logger = logging.getLogger(__name__)


class InjuryPreventionAgent(BaseAgent):
    """运动损伤预防Agent"""
    
    def __init__(self):
        super().__init__(
            name="InjuryPreventionAgent",
            description="识别运动损伤风险并提供预防措施"
        )
        self.risk_factors = self._initialize_risk_factors()
        self.prevention_strategies = self._initialize_prevention_strategies()
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化agent"""
        try:
            if config:
                if 'risk_factors' in config:
                    self.risk_factors.update(config['risk_factors'])
                if 'prevention_strategies' in config:
                    self.prevention_strategies.update(config['prevention_strategies'])
            self.is_initialized = True
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def process(self, user_context: UserContext, 
                body_analysis: Optional[Dict[str, Any]] = None,
                exercise_plan: Optional[Dict[str, Any]] = None,
                **kwargs) -> AgentResponse:
        """分析损伤风险并提供预防建议"""
        is_valid, error_msg = self.validate_input(user_context)
        if not is_valid:
            return AgentResponse(
                agent_name=self.name,
                success=False,
                data={},
                reasoning=f"Input validation failed: {error_msg}"
            )
        
        try:
            # 识别风险因素
            risk_assessment = self._assess_injury_risks(user_context, body_analysis, exercise_plan)
            
            # 生成预防措施
            prevention_measures = self._generate_prevention_measures(
                user_context, risk_assessment, exercise_plan
            )
            
            # 制定监控计划
            monitoring_plan = self._create_monitoring_plan(user_context, risk_assessment)
            
            # 推荐恢复策略
            recovery_strategies = self._recommend_recovery_strategies(user_context)
            
            # 生成建议
            recommendations = self._generate_recommendations(
                user_context, risk_assessment, prevention_measures
            )
            
            confidence = self._calculate_confidence(user_context, body_analysis, exercise_plan)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                data={
                    'risk_assessment': risk_assessment,
                    'prevention_measures': prevention_measures,
                    'monitoring_plan': monitoring_plan,
                    'recovery_strategies': recovery_strategies,
                    'warning_signs': self._identify_warning_signs(user_context)
                },
                confidence=confidence,
                reasoning=f"识别到{len(risk_assessment['high_risk_factors'])}个高风险因素，提供针对性预防措施",
                recommendations=recommendations
            )
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                data={},
                reasoning=f"Processing error: {str(e)}"
            )
    
    def _initialize_risk_factors(self) -> Dict[str, Dict[str, Any]]:
        """初始化风险因素数据库"""
        return {
            'previous_injury': {
                'weight': 0.3,
                'description': '既往损伤史是再次损伤的主要风险因素',
                'mitigation': '加强受伤部位的康复训练和预防性锻炼'
            },
            'overuse': {
                'weight': 0.25,
                'description': '训练负荷过大或恢复不足',
                'mitigation': '合理安排训练和休息时间'
            },
            'poor_form': {
                'weight': 0.2,
                'description': '运动姿势不正确',
                'mitigation': '学习正确的运动技巧，必要时寻求专业指导'
            },
            'muscle_imbalance': {
                'weight': 0.15,
                'description': '肌肉力量不平衡',
                'mitigation': '进行平衡性训练和核心力量训练'
            },
            'insufficient_warmup': {
                'weight': 0.1,
                'description': '热身不充分',
                'mitigation': '充分热身，包括动态拉伸'
            }
        }
    
    def _initialize_prevention_strategies(self) -> Dict[str, List[str]]:
        """初始化预防策略"""
        return {
            'warmup': [
                '5-10分钟轻度有氧运动',
                '动态拉伸主要运动肌群',
                '运动专项准备活动'
            ],
            'cool_down': [
                '5-10分钟低强度运动',
                '静态拉伸',
                '泡沫轴放松'
            ],
            'strength_training': [
                '加强核心力量',
                '平衡性训练',
                '功能性训练'
            ],
            'flexibility': [
                '定期进行柔韧性训练',
                '瑜伽或普拉提',
                '动态和静态拉伸结合'
            ],
            'load_management': [
                '渐进式增加训练负荷',
                '遵循10%原则（每周增加不超过10%）',
                '安排恢复日'
            ]
        }
    
    def _assess_injury_risks(self, user_context: UserContext,
                            body_analysis: Optional[Dict[str, Any]],
                            exercise_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """评估损伤风险"""
        risk_factors = []
        risk_score = 0.0
        
        # 既往损伤史
        if user_context.injury_history:
            for injury in user_context.injury_history:
                risk_factors.append({
                    'factor': 'previous_injury',
                    'severity': 'high',
                    'description': f"既往{injury.get('type', '损伤')}史",
                    'body_part': injury.get('body_part', 'unknown')
                })
                risk_score += self.risk_factors['previous_injury']['weight']
        
        # 健康条件
        if user_context.health_conditions:
            for condition in user_context.health_conditions:
                if any(keyword in condition.lower() for keyword in ['joint', 'bone', 'muscle', 'knee', 'back']):
                    risk_factors.append({
                        'factor': 'health_condition',
                        'severity': 'medium',
                        'description': f"健康问题: {condition}",
                        'body_part': condition
                    })
                    risk_score += 0.15
        
        # 身体分析相关风险
        if body_analysis:
            bmi_category = body_analysis.get('bmi_category', 'normal')
            if bmi_category == 'obese':
                risk_factors.append({
                    'factor': 'high_bmi',
                    'severity': 'medium',
                    'description': '高BMI增加关节负担',
                    'body_part': 'joints'
                })
                risk_score += 0.1
            
            fitness_assessment = body_analysis.get('fitness_assessment', {})
            if fitness_assessment.get('level') == 'beginner':
                risk_factors.append({
                    'factor': 'low_fitness',
                    'severity': 'medium',
                    'description': '健身水平较低，需要循序渐进',
                    'body_part': 'general'
                })
                risk_score += 0.1
        
        # 训练计划相关风险
        if exercise_plan:
            frequency = exercise_plan.get('weekly_frequency', 0)
            if frequency > 5:
                risk_factors.append({
                    'factor': 'high_frequency',
                    'severity': 'medium',
                    'description': '训练频率过高可能导致过度训练',
                    'body_part': 'general'
                })
                risk_score += 0.1
        
        # 年龄相关风险
        if user_context.age > 50:
            risk_factors.append({
                'factor': 'age',
                'severity': 'low',
                'description': '年龄增长需要更注意恢复和预防',
                'body_part': 'general'
            })
            risk_score += 0.05
        
        # 分类风险因素
        high_risk = [rf for rf in risk_factors if rf['severity'] == 'high']
        medium_risk = [rf for rf in risk_factors if rf['severity'] == 'medium']
        low_risk = [rf for rf in risk_factors if rf['severity'] == 'low']
        
        overall_risk_level = 'high' if risk_score > 0.5 else 'medium' if risk_score > 0.3 else 'low'
        
        return {
            'risk_score': min(risk_score, 1.0),
            'overall_risk_level': overall_risk_level,
            'high_risk_factors': high_risk,
            'medium_risk_factors': medium_risk,
            'low_risk_factors': low_risk,
            'all_factors': risk_factors
        }
    
    def _generate_prevention_measures(self, user_context: UserContext,
                                     risk_assessment: Dict[str, Any],
                                     exercise_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成预防措施"""
        measures = []
        
        # 针对高风险因素的措施
        for factor in risk_assessment['high_risk_factors']:
            factor_type = factor['factor']
            if factor_type in self.risk_factors:
                measures.append({
                    'priority': 'high',
                    'category': factor_type,
                    'measures': [self.risk_factors[factor_type]['mitigation']],
                    'target_body_part': factor.get('body_part', 'general')
                })
        
        # 通用预防措施
        measures.extend([
            {
                'priority': 'medium',
                'category': 'warmup',
                'measures': self.prevention_strategies['warmup'],
                'target_body_part': 'general'
            },
            {
                'priority': 'medium',
                'category': 'cool_down',
                'measures': self.prevention_strategies['cool_down'],
                'target_body_part': 'general'
            },
            {
                'priority': 'medium',
                'category': 'strength_training',
                'measures': self.prevention_strategies['strength_training'],
                'target_body_part': 'core'
            }
        ])
        
        # 如果有既往损伤，添加针对性措施
        if user_context.injury_history:
            for injury in user_context.injury_history:
                body_part = injury.get('body_part', 'unknown')
                measures.append({
                    'priority': 'high',
                    'category': 'rehabilitation',
                    'measures': [
                        f'加强{body_part}的康复训练',
                        f'进行{body_part}的预防性锻炼',
                        '定期评估恢复情况'
                    ],
                    'target_body_part': body_part
                })
        
        return measures
    
    def _create_monitoring_plan(self, user_context: UserContext,
                               risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """创建监控计划"""
        monitoring_frequency = 'daily' if risk_assessment['risk_score'] > 0.5 else 'weekly'
        
        plan = {
            'frequency': monitoring_frequency,
            'metrics_to_track': [
                'pain_level',
                'fatigue_level',
                'sleep_quality',
                'training_load',
                'recovery_time'
            ],
            'warning_thresholds': {
                'pain_level': 3,  # 0-10 scale
                'fatigue_level': 7,  # 0-10 scale
                'sleep_quality': 5  # 0-10 scale
            },
            'checkpoints': []
        }
        
        # 根据风险水平设置检查点
        if risk_assessment['risk_score'] > 0.5:
            plan['checkpoints'] = ['before_each_session', 'after_each_session', 'weekly_review']
        else:
            plan['checkpoints'] = ['before_each_session', 'weekly_review']
        
        return plan
    
    def _recommend_recovery_strategies(self, user_context: UserContext) -> List[Dict[str, Any]]:
        """推荐恢复策略"""
        strategies = [
            {
                'type': 'active_recovery',
                'description': '积极恢复',
                'activities': ['轻度散步', '瑜伽', '拉伸'],
                'frequency': '训练后或休息日'
            },
            {
                'type': 'sleep',
                'description': '充足睡眠',
                'recommendation': '每晚7-9小时高质量睡眠',
                'tips': ['保持规律作息', '创造良好睡眠环境']
            },
            {
                'type': 'nutrition',
                'description': '营养恢复',
                'recommendation': '训练后30分钟内补充蛋白质和碳水化合物',
                'tips': ['充足水分', '均衡饮食']
            },
            {
                'type': 'stress_management',
                'description': '压力管理',
                'activities': ['冥想', '深呼吸', '放松活动'],
                'frequency': '每日'
            }
        ]
        
        # 根据年龄调整
        if user_context.age > 50:
            strategies.append({
                'type': 'extended_recovery',
                'description': '延长恢复时间',
                'recommendation': '建议训练间隔至少48小时',
                'tips': ['更多休息日', '低强度活动']
            })
        
        return strategies
    
    def _identify_warning_signs(self, user_context: UserContext) -> List[Dict[str, Any]]:
        """识别警告信号"""
        warning_signs = [
            {
                'sign': '持续疼痛',
                'description': '运动后持续超过48小时的疼痛',
                'action': '减少训练强度或暂停训练，咨询医生'
            },
            {
                'sign': '关节肿胀',
                'description': '关节出现肿胀或僵硬',
                'action': '立即停止运动，冰敷，必要时就医'
            },
            {
                'sign': '活动范围受限',
                'description': '关节活动范围明显减少',
                'action': '停止相关运动，进行康复评估'
            },
            {
                'sign': '异常疲劳',
                'description': '持续疲劳，恢复时间延长',
                'action': '增加休息时间，检查训练负荷'
            },
            {
                'sign': '运动表现下降',
                'description': '运动能力明显下降',
                'action': '可能是过度训练，需要调整计划'
            }
        ]
        
        return warning_signs
    
    def _generate_recommendations(self, user_context: UserContext,
                                 risk_assessment: Dict[str, Any],
                                 prevention_measures: List[Dict[str, Any]]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if risk_assessment['risk_score'] > 0.5:
            recommendations.append("⚠️ 检测到较高的损伤风险，建议采取预防措施")
        
        if risk_assessment['high_risk_factors']:
            recommendations.append(f"重点关注{len(risk_assessment['high_risk_factors'])}个高风险因素")
        
        recommendations.append("训练前务必充分热身，训练后进行拉伸放松")
        recommendations.append("遵循渐进式训练原则，避免突然增加强度")
        
        if user_context.injury_history:
            recommendations.append("针对既往损伤部位进行预防性训练")
        
        recommendations.append("如出现疼痛或不适，立即停止运动并咨询专业人士")
        
        return recommendations
    
    def _calculate_confidence(self, user_context: UserContext,
                             body_analysis: Optional[Dict[str, Any]],
                             exercise_plan: Optional[Dict[str, Any]]) -> float:
        """计算置信度"""
        confidence = 0.8
        
        if body_analysis:
            confidence += 0.05
        if exercise_plan:
            confidence += 0.05
        if user_context.injury_history:
            confidence += 0.05
        
        return min(0.95, confidence)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回agent能力描述"""
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': [
                '损伤风险识别',
                '预防措施推荐',
                '训练负荷监控',
                '恢复策略建议',
                '警告信号识别',
                '个性化预防方案'
            ],
            'outputs': [
                'risk_assessment',
                'prevention_measures',
                'monitoring_plan',
                'recovery_strategies',
                'warning_signs'
            ]
        }

