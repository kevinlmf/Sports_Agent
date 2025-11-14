"""
多智能体系统使用示例
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import AgentOrchestrator, UserContext


def example_complete_analysis():
    """示例：完整的多智能体分析"""
    print("=" * 60)
    print("示例1: 完整的多智能体分析")
    print("=" * 60)
    
    # 初始化协调器
    orchestrator = AgentOrchestrator()
    
    # 创建用户上下文
    user = UserContext(
        user_id="user_001",
        age=28,
        gender="female",
        height=165.0,
        weight=58.0,
        fitness_level="beginner",
        health_conditions=[],
        injury_history=[],
        fitness_goals=["weight_loss", "flexibility"],
        preferences={"preferred_time": "evening"}
    )
    
    # 执行完整分析
    results = orchestrator.process_complete_analysis(user)
    
    # 显示关键结果
    print(f"\n用户: {user.user_id}")
    print(f"整体置信度: {results['overall_confidence']:.1%}")
    print(f"\n摘要:")
    summary = results['summary']
    print(f"  - BMI: {summary.get('bmi', 'N/A')}")
    print(f"  - 身体状态: {summary.get('body_status', 'N/A')}")
    print(f"  - 损伤风险: {summary.get('injury_risk_level', 'N/A')}")
    print(f"  - 幸福感评分: {summary.get('wellness_score', 0):.2f}/1.0")
    
    print(f"\n推荐运动:")
    for ex in summary.get('recommended_exercises', [])[:3]:
        print(f"  - {ex}")
    
    print(f"\n前5条建议:")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        print(f"  {i}. {rec}")


def example_single_agent():
    """示例：使用单个智能体"""
    print("\n" + "=" * 60)
    print("示例2: 单个智能体分析")
    print("=" * 60)
    
    orchestrator = AgentOrchestrator()
    
    user = UserContext(
        user_id="user_002",
        age=35,
        gender="male",
        height=180.0,
        weight=85.0,
        fitness_level="intermediate",
        health_conditions=["knee_pain"],
        injury_history=[
            {"type": "knee", "body_part": "knee", "severity": "moderate"}
        ],
        fitness_goals=["strength"],
        preferences={}
    )
    
    # 只使用身体分析智能体
    print("\n使用BodyAnalysisAgent分析身体情况...")
    response = orchestrator.process_single_agent("body_analysis", user)
    
    if response.success:
        data = response.data
        print(f"\n分析结果:")
        print(f"  BMI: {data.get('bmi', 'N/A'):.1f}")
        print(f"  分类: {data.get('bmi_category', 'N/A')}")
        print(f"  健康风险等级: {data.get('health_risks', {}).get('overall_risk_level', 'N/A')}")
        print(f"\n建议:")
        for rec in response.recommendations[:3]:
            print(f"  - {rec}")
    
    # 使用损伤预防智能体
    print("\n使用InjuryPreventionAgent分析损伤风险...")
    response = orchestrator.process_single_agent("injury_prevention", user)
    
    if response.success:
        data = response.data
        risk_assessment = data.get('risk_assessment', {})
        print(f"\n风险评估:")
        print(f"  风险评分: {risk_assessment.get('risk_score', 0):.2f}")
        print(f"  风险等级: {risk_assessment.get('overall_risk_level', 'N/A')}")
        print(f"  高风险因素数量: {len(risk_assessment.get('high_risk_factors', []))}")
        print(f"\n预防措施:")
        measures = data.get('prevention_measures', [])
        for measure in measures[:3]:
            print(f"  - {measure.get('category', 'N/A')}: {', '.join(measure.get('measures', [])[:2])}")


def example_wellness_focus():
    """示例：重点关注身心健康"""
    print("\n" + "=" * 60)
    print("示例3: 身心健康分析")
    print("=" * 60)
    
    orchestrator = AgentOrchestrator()
    
    user = UserContext(
        user_id="user_003",
        age=42,
        gender="female",
        height=160.0,
        weight=65.0,
        fitness_level="beginner",
        health_conditions=[],
        injury_history=[],
        fitness_goals=["stress_relief", "general_fitness"],
        preferences={"preferred_time": "morning"}
    )
    
    # 使用身心健康分析智能体
    response = orchestrator.process_single_agent("wellness_analysis", user)
    
    if response.success:
        data = response.data
        wellness = data.get('overall_wellness', {})
        
        print(f"\n身心健康评估:")
        print(f"  整体幸福感评分: {wellness.get('score', 0):.2f}/1.0")
        print(f"  水平: {wellness.get('level', 'N/A')}")
        print(f"  描述: {wellness.get('description', 'N/A')}")
        
        mental_health = data.get('mental_health', {})
        print(f"\n心理健康:")
        print(f"  评分: {mental_health.get('score', 0):.2f}/1.0")
        print(f"  水平: {mental_health.get('level', 'N/A')}")
        
        stress = data.get('stress_assessment', {})
        print(f"\n压力评估:")
        print(f"  压力水平: {stress.get('stress_level', 0):.2f}")
        print(f"  等级: {stress.get('level', 'N/A')}")
        
        print(f"\n运动益处:")
        benefits = data.get('exercise_benefits', {})
        for category, benefit_list in list(benefits.items())[:2]:
            print(f"  {category}:")
            for b in benefit_list[:2]:
                print(f"    - {b}")


if __name__ == "__main__":
    try:
        example_complete_analysis()
        example_single_agent()
        example_wellness_focus()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()

