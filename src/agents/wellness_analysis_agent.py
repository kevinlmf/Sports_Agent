"""
Wellness Analysis Agent
分析运动对于身心健康的影响，包括：
- 心理健康评估
- 压力管理
- 睡眠质量分析
- 整体幸福感评估
- 运动对心理健康的积极影响
"""

from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResponse, UserContext
import logging

logger = logging.getLogger(__name__)


class WellnessAnalysisAgent(BaseAgent):
    """身心健康分析Agent"""
    
    def __init__(self):
        super().__init__(
            name="WellnessAnalysisAgent",
            description="分析运动对身心健康的影响，评估整体幸福感"
        )
        self.wellness_indicators = self._initialize_wellness_indicators()
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化agent"""
        try:
            if config and 'wellness_indicators' in config:
                self.wellness_indicators.update(config['wellness_indicators'])
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
        """分析身心健康状况"""
        is_valid, error_msg = self.validate_input(user_context)
        if not is_valid:
            return AgentResponse(
                agent_name=self.name,
                success=False,
                data={},
                reasoning=f"Input validation failed: {error_msg}"
            )
        
        try:
            # 心理健康评估
            mental_health = self._assess_mental_health(user_context)
            
            # 压力水平评估
            stress_assessment = self._assess_stress_levels(user_context)
            
            # 睡眠质量分析
            sleep_analysis = self._analyze_sleep_quality(user_context)
            
            # 运动对心理健康的益处
            exercise_benefits = self._analyze_exercise_benefits(
                user_context, exercise_plan
            )
            
            # 整体幸福感评估
            overall_wellness = self._assess_overall_wellness(
                user_context, mental_health, stress_assessment, sleep_analysis
            )
            
            # 生成建议
            recommendations = self._generate_recommendations(
                user_context, mental_health, stress_assessment, sleep_analysis, exercise_benefits
            )
            
            confidence = self._calculate_confidence(user_context, body_analysis, exercise_plan)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                data={
                    'mental_health': mental_health,
                    'stress_assessment': stress_assessment,
                    'sleep_analysis': sleep_analysis,
                    'exercise_benefits': exercise_benefits,
                    'overall_wellness': overall_wellness,
                    'wellness_score': overall_wellness['score']
                },
                confidence=confidence,
                reasoning=f"整体幸福感评分: {overall_wellness['score']:.2f}/1.0，{overall_wellness['level']}水平",
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
    
    def _initialize_wellness_indicators(self) -> Dict[str, Dict[str, Any]]:
        """初始化健康指标"""
        return {
            'mood': {
                'factors': ['exercise_frequency', 'sleep_quality', 'stress_level'],
                'benefits': ['改善情绪', '减少焦虑', '提高自信心']
            },
            'stress': {
                'factors': ['workload', 'exercise', 'sleep'],
                'benefits': ['降低压力', '改善应对能力', '提高抗压能力']
            },
            'sleep': {
                'factors': ['exercise_timing', 'exercise_intensity', 'daily_routine'],
                'benefits': ['改善睡眠质量', '延长深度睡眠', '减少失眠']
            },
            'energy': {
                'factors': ['exercise_type', 'nutrition', 'recovery'],
                'benefits': ['提高能量水平', '减少疲劳', '改善日间功能']
            },
            'social': {
                'factors': ['group_exercise', 'sports_activities'],
                'benefits': ['增强社交', '改善人际关系', '提高归属感']
            }
        }
    
    def _assess_mental_health(self, user_context: UserContext) -> Dict[str, Any]:
        """评估心理健康"""
        # 基于用户信息和偏好进行简化评估
        mental_health_score = 0.7  # 基础分数
        
        # 如果有运动目标，说明有积极态度
        if user_context.fitness_goals:
            mental_health_score += 0.1
        
        # 如果有健康条件，可能影响心理健康
        if user_context.health_conditions:
            mental_health_score -= 0.1
        
        # 年龄因素
        if 20 <= user_context.age <= 40:
            mental_health_score += 0.05
        
        mental_health_score = max(0.0, min(1.0, mental_health_score))
        
        return {
            'score': mental_health_score,
            'level': 'good' if mental_health_score > 0.7 else 'fair' if mental_health_score > 0.5 else 'needs_attention',
            'indicators': {
                'mood': 'positive' if mental_health_score > 0.6 else 'neutral',
                'anxiety_level': 'low' if mental_health_score > 0.7 else 'moderate',
                'self_esteem': 'good' if mental_health_score > 0.65 else 'moderate'
            },
            'benefits_from_exercise': [
                '释放内啡肽，改善情绪',
                '减少焦虑和抑郁症状',
                '提高自信心和自我效能感',
                '增强认知功能'
            ]
        }
    
    def _assess_stress_levels(self, user_context: UserContext) -> Dict[str, Any]:
        """评估压力水平"""
        stress_score = 0.5  # 中等压力（0-1，越高压力越大）
        
        # 年龄相关压力
        if 30 <= user_context.age <= 50:
            stress_score += 0.1  # 这个年龄段通常压力较大
        
        # 健康条件可能增加压力
        if user_context.health_conditions:
            stress_score += 0.15
        
        stress_score = max(0.0, min(1.0, stress_score))
        
        return {
            'stress_level': stress_score,
            'level': 'low' if stress_score < 0.4 else 'moderate' if stress_score < 0.7 else 'high',
            'stressors': self._identify_stressors(user_context),
            'exercise_benefits': [
                '运动是有效的压力释放方式',
                '有氧运动可以降低皮质醇水平',
                '规律运动提高应对压力的能力',
                '运动后产生放松感'
            ],
            'recommended_activities': [
                '有氧运动（跑步、游泳、骑行）',
                '瑜伽和冥想',
                '户外运动接触自然',
                '团队运动增强社交'
            ]
        }
    
    def _identify_stressors(self, user_context: UserContext) -> List[str]:
        """识别压力源"""
        stressors = []
        
        if user_context.health_conditions:
            stressors.append('健康问题')
        
        if user_context.age >= 30:
            stressors.append('生活压力（工作、家庭）')
        
        return stressors if stressors else ['一般生活压力']
    
    def _analyze_sleep_quality(self, user_context: UserContext) -> Dict[str, Any]:
        """分析睡眠质量"""
        sleep_score = 0.7  # 基础分数
        
        # 年龄因素
        if user_context.age > 50:
            sleep_score -= 0.1  # 年龄增长可能影响睡眠
        
        # 健康条件
        if user_context.health_conditions:
            sleep_score -= 0.1
        
        sleep_score = max(0.0, min(1.0, sleep_score))
        
        return {
            'sleep_quality_score': sleep_score,
            'level': 'good' if sleep_score > 0.7 else 'fair' if sleep_score > 0.5 else 'poor',
            'estimated_hours': self._estimate_sleep_hours(user_context),
            'exercise_benefits': [
                '规律运动可以改善睡眠质量',
                '有氧运动有助于深度睡眠',
                '运动可以调节生物钟',
                '运动减少失眠症状'
            ],
            'recommendations': [
                '避免睡前2小时内剧烈运动',
                '早晨或下午运动有助于睡眠',
                '建立规律的睡眠时间表',
                '运动后充分放松'
            ]
        }
    
    def _estimate_sleep_hours(self, user_context: UserContext) -> Dict[str, int]:
        """估算睡眠需求"""
        # 根据年龄估算
        if user_context.age < 18:
            recommended = 8-10
        elif user_context.age < 65:
            recommended = 7-9
        else:
            recommended = 7-8
        
        return {
            'recommended_hours': recommended,
            'minimum_hours': recommended - 1,
            'optimal_hours': recommended + 1
        }
    
    def _analyze_exercise_benefits(self, user_context: UserContext,
                                  exercise_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析运动对身心健康的益处"""
        benefits = {
            'physical_benefits': [
                '改善心血管健康',
                '增强肌肉力量',
                '提高柔韧性',
                '改善身体成分',
                '增强免疫系统'
            ],
            'mental_benefits': [
                '释放内啡肽，产生愉悦感',
                '减少焦虑和抑郁',
                '提高自信心',
                '改善认知功能',
                '增强专注力'
            ],
            'emotional_benefits': [
                '改善情绪',
                '减少压力',
                '提高幸福感',
                '增强自我效能感',
                '改善自我形象'
            ],
            'social_benefits': [
                '增强社交互动',
                '建立支持网络',
                '提高团队合作能力',
                '增强归属感'
            ]
        }
        
        # 根据运动计划评估具体益处
        if exercise_plan:
            exercise_types = exercise_plan.get('exercise_types', [])
            for ex_type in exercise_types:
                category = ex_type.get('category', '')
                if category == 'cardio':
                    benefits['specific_benefits'] = benefits.get('specific_benefits', [])
                    benefits['specific_benefits'].extend([
                        '有氧运动特别有助于心血管健康和压力释放',
                        '提高耐力和能量水平'
                    ])
                elif category == 'strength':
                    benefits['specific_benefits'] = benefits.get('specific_benefits', [])
                    benefits['specific_benefits'].extend([
                        '力量训练增强自信心和身体形象',
                        '改善骨密度和代谢健康'
                    ])
                elif category == 'flexibility':
                    benefits['specific_benefits'] = benefits.get('specific_benefits', [])
                    benefits['specific_benefits'].extend([
                        '柔韧性训练有助于放松和压力管理',
                        '改善身体意识和正念'
                    ])
        
        # 评估预期改善
        benefits['expected_improvements'] = {
            'short_term': [
                '运动后立即感到更轻松',
                '改善睡眠质量',
                '增加能量水平'
            ],
            'medium_term': [
                '持续改善情绪',
                '减少压力水平',
                '提高自信心'
            ],
            'long_term': [
                '整体幸福感提升',
                '更好的心理健康',
                '改善生活质量'
            ]
        }
        
        return benefits
    
    def _assess_overall_wellness(self, user_context: UserContext,
                                mental_health: Dict[str, Any],
                                stress_assessment: Dict[str, Any],
                                sleep_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """评估整体幸福感"""
        # 综合评分
        wellness_score = (
            mental_health['score'] * 0.4 +
            (1 - stress_assessment['stress_level']) * 0.3 +  # 压力越低越好
            sleep_analysis['sleep_quality_score'] * 0.3
        )
        
        wellness_score = max(0.0, min(1.0, wellness_score))
        
        if wellness_score >= 0.75:
            level = 'excellent'
            description = '整体健康状况优秀'
        elif wellness_score >= 0.6:
            level = 'good'
            description = '整体健康状况良好'
        elif wellness_score >= 0.45:
            level = 'fair'
            description = '整体健康状况一般，有改善空间'
        else:
            level = 'needs_improvement'
            description = '需要关注身心健康，建议采取行动'
        
        return {
            'score': wellness_score,
            'level': level,
            'description': description,
            'components': {
                'mental_health': mental_health['score'],
                'stress_management': 1 - stress_assessment['stress_level'],
                'sleep_quality': sleep_analysis['sleep_quality_score']
            },
            'trend': 'improving' if wellness_score > 0.6 else 'stable'
        }
    
    def _generate_recommendations(self, user_context: UserContext,
                                 mental_health: Dict[str, Any],
                                 stress_assessment: Dict[str, Any],
                                 sleep_analysis: Dict[str, Any],
                                 exercise_benefits: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于心理健康
        if mental_health['score'] < 0.6:
            recommendations.append("建议通过规律运动改善心理健康，运动可以释放内啡肽，提升情绪")
        
        # 基于压力水平
        if stress_assessment['stress_level'] > 0.6:
            recommendations.append("压力水平较高，建议进行有氧运动或瑜伽来缓解压力")
            recommendations.append("考虑户外运动，接触自然有助于降低压力")
        
        # 基于睡眠质量
        if sleep_analysis['sleep_quality_score'] < 0.6:
            recommendations.append("睡眠质量有待改善，规律运动可以帮助改善睡眠")
            recommendations.append("建议避免睡前2小时内进行剧烈运动")
        
        # 通用建议
        recommendations.append("规律运动对身心健康有显著益处，建议坚持运动")
        recommendations.append("运动不仅改善身体健康，还能提升心理健康和幸福感")
        
        # 社交建议
        if user_context.fitness_level.lower() != 'advanced':
            recommendations.append("考虑参加团体运动或健身课程，增强社交互动")
        
        return recommendations
    
    def _calculate_confidence(self, user_context: UserContext,
                             body_analysis: Optional[Dict[str, Any]],
                             exercise_plan: Optional[Dict[str, Any]]) -> float:
        """计算置信度"""
        confidence = 0.75
        
        if body_analysis:
            confidence += 0.1
        if exercise_plan:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回agent能力描述"""
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': [
                '心理健康评估',
                '压力水平分析',
                '睡眠质量评估',
                '运动益处分析',
                '整体幸福感评估',
                '身心健康建议'
            ],
            'outputs': [
                'mental_health',
                'stress_assessment',
                'sleep_analysis',
                'exercise_benefits',
                'overall_wellness',
                'wellness_score'
            ]
        }

