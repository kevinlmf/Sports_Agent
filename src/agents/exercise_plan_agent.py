"""
Exercise Plan Agent
负责选择最好的运动方案，包括：
- 根据用户情况推荐运动类型
- 制定个性化训练计划
- 优化运动强度和频率
- 考虑用户目标和偏好
"""

from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResponse, UserContext
import logging

logger = logging.getLogger(__name__)


class ExercisePlanAgent(BaseAgent):
    """运动方案Agent"""
    
    def __init__(self):
        super().__init__(
            name="ExercisePlanAgent",
            description="根据用户身体情况和目标，选择最佳运动方案"
        )
        self.exercise_database = self._initialize_exercise_database()
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化agent"""
        try:
            if config and 'exercise_database' in config:
                self.exercise_database.update(config['exercise_database'])
            self.is_initialized = True
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def process(self, user_context: UserContext, body_analysis: Optional[Dict[str, Any]] = None, **kwargs) -> AgentResponse:
        """生成运动方案"""
        is_valid, error_msg = self.validate_input(user_context)
        if not is_valid:
            return AgentResponse(
                agent_name=self.name,
                success=False,
                data={},
                reasoning=f"Input validation failed: {error_msg}"
            )
        
        try:
            # 分析用户需求
            user_needs = self._analyze_user_needs(user_context, body_analysis)
            
            # 选择适合的运动类型
            recommended_exercises = self._select_exercises(user_context, user_needs)
            
            # 制定训练计划
            training_plan = self._create_training_plan(user_context, recommended_exercises)
            
            # 优化计划
            optimized_plan = self._optimize_plan(training_plan, user_context)
            
            # 生成建议
            recommendations = self._generate_recommendations(user_context, optimized_plan)
            
            confidence = self._calculate_confidence(user_context, body_analysis)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                data={
                    'user_needs': user_needs,
                    'recommended_exercises': recommended_exercises,
                    'training_plan': optimized_plan,
                    'weekly_schedule': self._create_weekly_schedule(optimized_plan),
                    'progression_plan': self._create_progression_plan(user_context)
                },
                confidence=confidence,
                reasoning=f"基于用户目标{user_context.fitness_goals}和{user_context.fitness_level}水平制定计划",
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
    
    def _initialize_exercise_database(self) -> Dict[str, Dict[str, Any]]:
        """初始化运动数据库"""
        return {
            'cardio': {
                'name': '有氧运动',
                'types': ['running', 'cycling', 'swimming', 'walking', 'dancing'],
                'benefits': ['心血管健康', '减脂', '提高耐力'],
                'intensity_levels': ['low', 'moderate', 'high'],
                'suitable_for': ['weight_loss', 'endurance', 'general_fitness']
            },
            'strength': {
                'name': '力量训练',
                'types': ['weight_lifting', 'bodyweight', 'resistance_bands'],
                'benefits': ['增肌', '提高骨密度', '改善代谢'],
                'intensity_levels': ['low', 'moderate', 'high'],
                'suitable_for': ['muscle_gain', 'strength', 'bone_health']
            },
            'flexibility': {
                'name': '柔韧性训练',
                'types': ['yoga', 'stretching', 'pilates', 'tai_chi'],
                'benefits': ['改善柔韧性', '减少肌肉紧张', '放松身心'],
                'intensity_levels': ['low', 'moderate'],
                'suitable_for': ['flexibility', 'recovery', 'stress_relief']
            },
            'balance': {
                'name': '平衡训练',
                'types': ['balance_exercises', 'core_strengthening'],
                'benefits': ['改善平衡', '预防跌倒', '核心力量'],
                'intensity_levels': ['low', 'moderate'],
                'suitable_for': ['balance', 'injury_prevention', 'core_strength']
            }
        }
    
    def _analyze_user_needs(self, user_context: UserContext, 
                           body_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析用户需求"""
        needs = {
            'primary_goals': user_context.fitness_goals or ['general_fitness'],
            'fitness_level': user_context.fitness_level,
            'constraints': [],
            'preferences': user_context.preferences or {}
        }
        
        # 基于身体分析调整需求
        if body_analysis:
            bmi_category = body_analysis.get('bmi_category', 'normal')
            if bmi_category == 'overweight' or bmi_category == 'obese':
                if 'weight_loss' not in needs['primary_goals']:
                    needs['primary_goals'].append('weight_loss')
            elif bmi_category == 'underweight':
                if 'muscle_gain' not in needs['primary_goals']:
                    needs['primary_goals'].append('muscle_gain')
            
            health_risks = body_analysis.get('health_risks', {})
            if health_risks.get('risk_score', 0) > 0.5:
                needs['constraints'].append('high_risk')
        
        # 考虑健康条件限制
        if user_context.health_conditions:
            needs['constraints'].extend(user_context.health_conditions)
        
        return needs
    
    def _select_exercises(self, user_context: UserContext, 
                         user_needs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """选择适合的运动"""
        selected_exercises = []
        goals = user_needs['primary_goals']
        fitness_level = user_needs['fitness_level']
        
        # 根据目标选择运动类型
        for goal in goals:
            for category, info in self.exercise_database.items():
                if goal in info['suitable_for']:
                    # 根据健身水平选择强度
                    if fitness_level == 'beginner':
                        intensity = 'low'
                    elif fitness_level == 'intermediate':
                        intensity = 'moderate'
                    else:
                        intensity = 'high'
                    
                    selected_exercises.append({
                        'category': category,
                        'name': info['name'],
                        'types': info['types'][:2],  # 选择前2种类型
                        'intensity': intensity,
                        'benefits': info['benefits'],
                        'reason': f"适合{goal}目标"
                    })
        
        # 去重
        seen_categories = set()
        unique_exercises = []
        for ex in selected_exercises:
            if ex['category'] not in seen_categories:
                unique_exercises.append(ex)
                seen_categories.add(ex['category'])
        
        # 如果没有选择，提供通用方案
        if not unique_exercises:
            unique_exercises.append({
                'category': 'cardio',
                'name': '有氧运动',
                'types': ['walking', 'cycling'],
                'intensity': 'moderate',
                'benefits': ['整体健康', '心血管健康'],
                'reason': '通用健康方案'
            })
        
        return unique_exercises
    
    def _create_training_plan(self, user_context: UserContext, 
                            exercises: List[Dict[str, Any]]) -> Dict[str, Any]:
        """制定训练计划"""
        fitness_level = user_context.fitness_level.lower()
        
        # 根据健身水平确定频率和时长
        if fitness_level == 'beginner':
            frequency = 3  # 每周3次
            duration = 30  # 每次30分钟
        elif fitness_level == 'intermediate':
            frequency = 4  # 每周4次
            duration = 45  # 每次45分钟
        else:
            frequency = 5  # 每周5次
            duration = 60  # 每次60分钟
        
        plan = {
            'weekly_frequency': frequency,
            'session_duration_minutes': duration,
            'exercise_types': exercises,
            'warm_up': {
                'duration': 5,
                'exercises': ['light_jogging', 'dynamic_stretching']
            },
            'cool_down': {
                'duration': 5,
                'exercises': ['static_stretching', 'breathing']
            }
        }
        
        return plan
    
    def _optimize_plan(self, plan: Dict[str, Any], 
                      user_context: UserContext) -> Dict[str, Any]:
        """优化训练计划"""
        optimized = plan.copy()
        
        # 考虑用户偏好
        if user_context.preferences:
            preferred_time = user_context.preferences.get('preferred_time', 'morning')
            optimized['preferred_time'] = preferred_time
            
            equipment = user_context.preferences.get('equipment', 'none')
            optimized['equipment_needed'] = equipment
        
        # 考虑健康限制
        if user_context.health_conditions:
            # 如果有膝盖问题，避免高冲击运动
            if any('knee' in cond.lower() for cond in user_context.health_conditions):
                optimized['modifications'] = ['low_impact_exercises']
        
        return optimized
    
    def _create_weekly_schedule(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建周计划"""
        frequency = plan['weekly_frequency']
        exercises = plan['exercise_types']
        
        # 简单的周计划分配
        schedule = []
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i in range(frequency):
            day_idx = i % len(days)
            exercise_idx = i % len(exercises)
            
            schedule.append({
                'day': days[day_idx],
                'exercise_category': exercises[exercise_idx]['category'],
                'exercise_types': exercises[exercise_idx]['types'],
                'duration_minutes': plan['session_duration_minutes'],
                'intensity': exercises[exercise_idx]['intensity']
            })
        
        return schedule
    
    def _create_progression_plan(self, user_context: UserContext) -> Dict[str, Any]:
        """创建进阶计划"""
        fitness_level = user_context.fitness_level.lower()
        
        progression = {
            'current_level': fitness_level,
            'weeks': []
        }
        
        # 4周进阶计划
        for week in range(1, 5):
            if fitness_level == 'beginner':
                progression['weeks'].append({
                    'week': week,
                    'focus': f'适应和建立基础' if week <= 2 else '逐步增加强度',
                    'intensity_increase': 0.1 if week > 2 else 0.0
                })
            elif fitness_level == 'intermediate':
                progression['weeks'].append({
                    'week': week,
                    'focus': f'优化训练效果',
                    'intensity_increase': 0.05 * week
                })
            else:
                progression['weeks'].append({
                    'week': week,
                    'focus': f'挑战和突破',
                    'intensity_increase': 0.1 * week
                })
        
        return progression
    
    def _generate_recommendations(self, user_context: UserContext, 
                                 plan: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        recommendations.append(f"建议每周进行{plan['weekly_frequency']}次训练，每次{plan['session_duration_minutes']}分钟")
        recommendations.append("训练前务必进行5-10分钟热身，训练后进行拉伸放松")
        
        if user_context.fitness_level.lower() == 'beginner':
            recommendations.append("建议从低强度开始，逐步适应后再增加强度")
        
        if user_context.health_conditions:
            recommendations.append("如有不适，请立即停止运动并咨询医生")
        
        return recommendations
    
    def _calculate_confidence(self, user_context: UserContext, 
                             body_analysis: Optional[Dict[str, Any]]) -> float:
        """计算置信度"""
        confidence = 0.75
        
        if body_analysis:
            confidence += 0.1
        
        if user_context.fitness_goals:
            confidence += 0.05
        
        if user_context.preferences:
            confidence += 0.05
        
        return min(0.95, confidence)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回agent能力描述"""
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': [
                '个性化运动方案推荐',
                '训练计划制定',
                '周计划安排',
                '进阶计划设计',
                '运动类型选择',
                '强度优化'
            ],
            'outputs': [
                'user_needs',
                'recommended_exercises',
                'training_plan',
                'weekly_schedule',
                'progression_plan'
            ]
        }

