"""
测试多智能体系统
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import AgentOrchestrator, UserContext

def test_multi_agent_system():
    """测试多智能体系统"""
    print("=" * 60)
    print("多智能体运动健康管理系统测试")
    print("=" * 60)
    
    # 初始化协调器
    print("\n1. 初始化Agent协调器...")
    orchestrator = AgentOrchestrator()
    print("✓ 协调器初始化成功")
    
    # 创建测试用户上下文
    print("\n2. 创建测试用户上下文...")
    user_context = UserContext(
        user_id="test_user_001",
        age=30,
        gender="male",
        height=175.0,
        weight=75.0,
        fitness_level="intermediate",
        health_conditions=[],
        injury_history=[
            {"type": "knee", "body_part": "knee", "severity": "minor"}
        ],
        fitness_goals=["weight_loss", "muscle_gain"],
        preferences={"preferred_time": "morning", "equipment": "gym"}
    )
    print("✓ 用户上下文创建成功")
    print(f"  用户ID: {user_context.user_id}")
    print(f"  年龄: {user_context.age}岁")
    print(f"  身高: {user_context.height}cm")
    print(f"  体重: {user_context.weight}kg")
    print(f"  健身水平: {user_context.fitness_level}")
    
    # 执行完整分析
    print("\n3. 执行完整的多智能体分析...")
    print("-" * 60)
    results = orchestrator.process_complete_analysis(user_context)
    
    if results.get('error'):
        print(f"✗ 分析失败: {results['error']}")
        return
    
    print("✓ 分析完成")
    print(f"\n工作流ID: {results['workflow_id']}")
    print(f"整体置信度: {results['overall_confidence']:.2%}")
    
    # 显示摘要
    print("\n4. 分析摘要:")
    print("-" * 60)
    summary = results['summary']
    print(f"身体状态: {summary.get('body_status', 'unknown')}")
    if 'bmi' in summary:
        print(f"BMI: {summary['bmi']:.1f}")
    print(f"推荐运动类型: {', '.join(summary.get('recommended_exercises', []))}")
    print(f"损伤风险等级: {summary.get('injury_risk_level', 'unknown')}")
    print(f"幸福感评分: {summary.get('wellness_score', 0):.2f}/1.0")
    
    # 显示各智能体结果
    print("\n5. 各智能体分析结果:")
    print("-" * 60)
    for agent_name, agent_result in results['agents'].items():
        print(f"\n{agent_name}:")
        print(f"  成功: {'✓' if agent_result['success'] else '✗'}")
        print(f"  置信度: {agent_result['confidence']:.2%}")
        print(f"  推理: {agent_result['reasoning']}")
        if agent_result['recommendations']:
            print(f"  建议:")
            for rec in agent_result['recommendations'][:3]:  # 只显示前3条
                print(f"    - {rec}")
    
    # 显示综合建议
    print("\n6. 综合建议:")
    print("-" * 60)
    for i, rec in enumerate(results['recommendations'][:10], 1):  # 显示前10条
        print(f"{i}. {rec}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


def test_single_agent():
    """测试单个智能体"""
    print("\n" + "=" * 60)
    print("单个智能体测试")
    print("=" * 60)
    
    orchestrator = AgentOrchestrator()
    
    user_context = UserContext(
        user_id="test_user_002",
        age=25,
        gender="female",
        height=165.0,
        weight=60.0,
        fitness_level="beginner",
        health_conditions=[],
        injury_history=[],
        fitness_goals=["general_fitness"],
        preferences={}
    )
    
    print("\n测试BodyAnalysisAgent...")
    response = orchestrator.process_single_agent("body_analysis", user_context)
    
    print(f"智能体: {response.agent_name}")
    print(f"成功: {'✓' if response.success else '✗'}")
    print(f"置信度: {response.confidence:.2%}")
    
    if response.success:
        data = response.data
        print(f"\n分析结果:")
        print(f"  BMI: {data.get('bmi', 'N/A')}")
        print(f"  BMI分类: {data.get('bmi_category', 'N/A')}")
        print(f"  健康风险等级: {data.get('health_risks', {}).get('overall_risk_level', 'N/A')}")
        print(f"\n建议:")
        for rec in response.recommendations[:5]:
            print(f"  - {rec}")


if __name__ == "__main__":
    try:
        test_multi_agent_system()
        test_single_agent()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

