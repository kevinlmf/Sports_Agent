"""
Body Analysis Agent
负责分析用户身体情况，包括：
- 身体指标评估（BMI、体脂率、肌肉量等）
- 健康风险评估
- 身体机能分析
- 营养状态评估
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .base_agent import BaseAgent, AgentResponse, UserContext
import logging

logger = logging.getLogger(__name__)


class BodyAnalysisAgent(BaseAgent):
    """身体分析Agent"""
    
    def __init__(self):
        super().__init__(
            name="BodyAnalysisAgent",
            description="分析用户身体情况，评估健康指标和风险"
        )
        self.bmi_thresholds = {
            'underweight': 18.5,
            'normal': 25.0,
            'overweight': 30.0
        }
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化agent"""
        try:
            if config:
                if 'bmi_thresholds' in config:
                    self.bmi_thresholds.update(config['bmi_thresholds'])
            self.is_initialized = True
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def process(self, user_context: UserContext, **kwargs) -> AgentResponse:
        """分析用户身体情况"""
        # 验证输入
        is_valid, error_msg = self.validate_input(user_context)
        if not is_valid:
            return AgentResponse(
                agent_name=self.name,
                success=False,
                data={},
                reasoning=f"Input validation failed: {error_msg}"
            )
        
        try:
            # 计算BMI
            bmi = self._calculate_bmi(user_context.height, user_context.weight)
            bmi_category = self._categorize_bmi(bmi)
            
            # 评估健康风险
            health_risks = self._assess_health_risks(user_context, bmi)
            
            # 分析身体机能
            fitness_assessment = self._assess_fitness_level(user_context)
            
            # 营养状态评估
            nutrition_status = self._assess_nutrition(user_context, bmi)
            
            # 生成建议
            recommendations = self._generate_recommendations(
                user_context, bmi, health_risks, fitness_assessment
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(user_context)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                data={
                    'bmi': round(bmi, 2),
                    'bmi_category': bmi_category,
                    'health_risks': health_risks,
                    'fitness_assessment': fitness_assessment,
                    'nutrition_status': nutrition_status,
                    'body_composition': self._estimate_body_composition(user_context),
                    'metabolic_health': self._assess_metabolic_health(user_context, bmi)
                },
                confidence=confidence,
                reasoning=f"基于用户年龄{user_context.age}岁，BMI {bmi:.1f}，评估为{bmi_category}",
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
    
    def _calculate_bmi(self, height_cm: float, weight_kg: float) -> float:
        """计算BMI"""
        height_m = height_cm / 100.0
        return weight_kg / (height_m ** 2)
    
    def _categorize_bmi(self, bmi: float) -> str:
        """BMI分类"""
        if bmi < self.bmi_thresholds['underweight']:
            return 'underweight'
        elif bmi < self.bmi_thresholds['normal']:
            return 'normal'
        elif bmi < self.bmi_thresholds['overweight']:
            return 'overweight'
        else:
            return 'obese'
    
    def _assess_health_risks(self, user_context: UserContext, bmi: float) -> Dict[str, Any]:
        """评估健康风险"""
        risks = []
        risk_score = 0.0
        
        # BMI相关风险
        if bmi >= 30:
            risks.append({
                'type': 'obesity',
                'severity': 'high',
                'description': '肥胖可能增加心血管疾病、糖尿病等风险'
            })
            risk_score += 0.3
        elif bmi < 18.5:
            risks.append({
                'type': 'underweight',
                'severity': 'medium',
                'description': '体重过轻可能影响免疫力和骨密度'
            })
            risk_score += 0.2
        
        # 年龄相关风险
        if user_context.age > 50:
            risks.append({
                'type': 'age_related',
                'severity': 'medium',
                'description': '年龄增长需要更关注心血管和骨骼健康'
            })
            risk_score += 0.1
        
        # 健康条件风险
        if user_context.health_conditions:
            for condition in user_context.health_conditions:
                risks.append({
                    'type': 'health_condition',
                    'severity': 'medium',
                    'description': f'存在健康问题: {condition}'
                })
                risk_score += 0.15
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risks': risks,
            'overall_risk_level': 'high' if risk_score > 0.5 else 'medium' if risk_score > 0.3 else 'low'
        }
    
    def _assess_fitness_level(self, user_context: UserContext) -> Dict[str, Any]:
        """评估身体机能水平"""
        fitness_level_map = {
            'beginner': {'score': 0.3, 'description': '初学者，建议从低强度运动开始'},
            'intermediate': {'score': 0.6, 'description': '中等水平，可以尝试中等强度运动'},
            'advanced': {'score': 0.9, 'description': '高级水平，可以进行高强度训练'}
        }
        
        level_info = fitness_level_map.get(
            user_context.fitness_level.lower(),
            fitness_level_map['beginner']
        )
        
        return {
            'level': user_context.fitness_level,
            'score': level_info['score'],
            'description': level_info['description'],
            'estimated_vo2_max': self._estimate_vo2_max(user_context)
        }
    
    def _estimate_vo2_max(self, user_context: UserContext) -> float:
        """估算最大摄氧量（简化版）"""
        # 基于年龄、性别、BMI的简化估算
        base_vo2 = 50.0
        age_factor = -0.3 * (user_context.age - 20)
        bmi_factor = -0.5 * (self._calculate_bmi(user_context.height, user_context.weight) - 22)
        
        if user_context.gender.lower() == 'female':
            base_vo2 -= 5.0
        
        estimated = base_vo2 + age_factor + bmi_factor
        return max(20.0, min(60.0, estimated))
    
    def _assess_nutrition(self, user_context: UserContext, bmi: float) -> Dict[str, Any]:
        """营养状态评估"""
        status = 'normal'
        if bmi < 18.5:
            status = 'underweight'
        elif bmi >= 30:
            status = 'overweight'
        
        return {
            'status': status,
            'estimated_daily_calories': self._estimate_daily_calories(user_context),
            'protein_needs': self._estimate_protein_needs(user_context)
        }
    
    def _estimate_daily_calories(self, user_context: UserContext) -> Dict[str, float]:
        """估算每日卡路里需求"""
        # BMR (Basal Metabolic Rate) 使用Mifflin-St Jeor公式
        if user_context.gender.lower() == 'male':
            bmr = 10 * user_context.weight + 6.25 * user_context.height - 5 * user_context.age + 5
        else:
            bmr = 10 * user_context.weight + 6.25 * user_context.height - 5 * user_context.age - 161
        
        # 活动系数
        activity_multipliers = {
            'beginner': 1.375,
            'intermediate': 1.55,
            'advanced': 1.725
        }
        multiplier = activity_multipliers.get(user_context.fitness_level.lower(), 1.5)
        
        maintenance = bmr * multiplier
        
        return {
            'bmr': round(bmr, 0),
            'maintenance': round(maintenance, 0),
            'weight_loss': round(maintenance * 0.85, 0),
            'weight_gain': round(maintenance * 1.15, 0)
        }
    
    def _estimate_protein_needs(self, user_context: UserContext) -> float:
        """估算蛋白质需求（克/天）"""
        # 一般建议：1.2-2.0g/kg体重
        protein_per_kg = 1.5
        if user_context.fitness_level.lower() == 'advanced':
            protein_per_kg = 1.8
        
        return round(user_context.weight * protein_per_kg, 1)
    
    def _estimate_body_composition(self, user_context: UserContext) -> Dict[str, float]:
        """估算身体成分（简化版）"""
        bmi = self._calculate_bmi(user_context.height, user_context.weight)
        
        # 简化的体脂率估算（基于BMI和年龄）
        if user_context.gender.lower() == 'male':
            body_fat = 1.20 * bmi + 0.23 * user_context.age - 16.2
        else:
            body_fat = 1.20 * bmi + 0.23 * user_context.age - 5.4
        
        body_fat = max(5.0, min(50.0, body_fat))
        lean_mass = user_context.weight * (1 - body_fat / 100)
        
        return {
            'body_fat_percentage': round(body_fat, 1),
            'lean_mass_kg': round(lean_mass, 1),
            'fat_mass_kg': round(user_context.weight - lean_mass, 1)
        }
    
    def _assess_metabolic_health(self, user_context: UserContext, bmi: float) -> Dict[str, Any]:
        """评估代谢健康"""
        metabolic_score = 0.7  # 基础分数
        
        if 18.5 <= bmi <= 25:
            metabolic_score += 0.2
        elif bmi > 30:
            metabolic_score -= 0.2
        
        if user_context.age < 40:
            metabolic_score += 0.1
        
        return {
            'score': max(0.0, min(1.0, metabolic_score)),
            'status': 'good' if metabolic_score > 0.7 else 'fair' if metabolic_score > 0.5 else 'poor'
        }
    
    def _generate_recommendations(self, user_context: UserContext, bmi: float, 
                                 health_risks: Dict, fitness_assessment: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if bmi < 18.5:
            recommendations.append("建议增加营养摄入，进行力量训练以增加肌肉量")
        elif bmi >= 30:
            recommendations.append("建议通过饮食控制和有氧运动来降低体重")
            recommendations.append("建议咨询营养师制定个性化饮食计划")
        
        if user_context.fitness_level.lower() == 'beginner':
            recommendations.append("建议从低强度运动开始，逐步增加运动量")
        
        if health_risks['risk_score'] > 0.5:
            recommendations.append("建议定期进行健康体检，关注心血管健康")
        
        if not recommendations:
            recommendations.append("保持当前的健康生活方式，定期运动")
        
        return recommendations
    
    def _calculate_confidence(self, user_context: UserContext) -> float:
        """计算分析置信度"""
        confidence = 0.8  # 基础置信度
        
        # 如果有更多信息，置信度更高
        if user_context.health_conditions:
            confidence += 0.1
        if user_context.injury_history:
            confidence += 0.05
        
        return min(0.95, confidence)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """返回agent能力描述"""
        return {
            'name': self.name,
            'description': self.description,
            'capabilities': [
                'BMI计算和分类',
                '健康风险评估',
                '身体机能评估',
                '营养状态分析',
                '身体成分估算',
                '代谢健康评估'
            ],
            'outputs': [
                'bmi',
                'bmi_category',
                'health_risks',
                'fitness_assessment',
                'nutrition_status',
                'body_composition',
                'metabolic_health'
            ]
        }

