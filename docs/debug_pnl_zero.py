"""
Diagnostic Script for Debugging Zero PnL in RL Market Making

This script provides diagnostic functions to identify why PnL stays at $0
during SAC training for market making.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def diagnose_actions(actions: np.ndarray, action_space_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose action outputs to check if they're valid.
    
    Args:
        actions: Array of actions from agent
        action_space_info: Dictionary with action space bounds
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'action_mean': float(np.mean(actions)),
        'action_std': float(np.std(actions)),
        'action_min': float(np.min(actions)),
        'action_max': float(np.max(actions)),
        'zero_actions_ratio': float(np.sum(actions == 0) / len(actions)),
        'actions_in_range': True,
        'actions_valid': True,
        'issues': []
    }
    
    # Check if actions are in valid range
    if 'low' in action_space_info and 'high' in action_space_info:
        low = action_space_info['low']
        high = action_space_info['high']
        out_of_range = np.any((actions < low) | (actions > high))
        diagnostics['actions_in_range'] = not out_of_range
        if out_of_range:
            diagnostics['issues'].append("Actions outside valid range")
    
    # Check for all-zero actions
    if diagnostics['zero_actions_ratio'] > 0.9:
        diagnostics['issues'].append(f"Too many zero actions: {diagnostics['zero_actions_ratio']:.2%}")
        diagnostics['actions_valid'] = False
    
    # Check for constant actions (no exploration)
    if diagnostics['action_std'] < 0.01:
        diagnostics['issues'].append("Actions have very low variance (no exploration)")
        diagnostics['actions_valid'] = False
    
    return diagnostics


def diagnose_environment_state(env, info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose environment state to check position tracking and PnL calculation.
    
    Args:
        env: Environment instance
        info: Info dictionary from step function
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'position': getattr(env, 'position', None),
        'realized_pnl': info.get('realized_pnl', None),
        'unrealized_pnl': info.get('unrealized_pnl', None),
        'total_pnl': info.get('pnl', None),
        'trades_executed': info.get('trades', 0),
        'orders_placed': info.get('orders_placed', 0),
        'orders_filled': info.get('orders_filled', 0),
        'issues': []
    }
    
    # Check if position is tracked
    if diagnostics['position'] is None:
        diagnostics['issues'].append("Position not tracked in environment")
    
    # Check if PnL components are available
    if diagnostics['realized_pnl'] is None and diagnostics['unrealized_pnl'] is None:
        diagnostics['issues'].append("PnL components not available in info")
    
    # Check if trades are being executed
    if diagnostics['orders_placed'] > 0:
        fill_rate = diagnostics['orders_filled'] / diagnostics['orders_placed']
        if fill_rate < 0.1:
            diagnostics['issues'].append(f"Low fill rate: {fill_rate:.2%}")
    else:
        diagnostics['issues'].append("No orders being placed")
    
    # Check if position changes
    if hasattr(env, 'position_history') and len(env.position_history) > 1:
        position_changes = np.diff(env.position_history[-10:])
        if np.all(position_changes == 0):
            diagnostics['issues'].append("Position not changing (stuck at same value)")
    
    return diagnostics


def diagnose_reward_function(env, action: np.ndarray, reward: float) -> Dict[str, Any]:
    """
    Diagnose reward function to check if PnL is included.
    
    Args:
        env: Environment instance
        action: Action taken
        reward: Reward received
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'reward': float(reward),
        'reward_components': {},
        'pnl_in_reward': False,
        'issues': []
    }
    
    # Try to get reward breakdown
    if hasattr(env, 'get_reward_breakdown'):
        diagnostics['reward_components'] = env.get_reward_breakdown()
        diagnostics['pnl_in_reward'] = 'pnl' in diagnostics['reward_components']
    else:
        diagnostics['issues'].append("Cannot access reward breakdown")
    
    # Check if reward is always the same (suggests no PnL component)
    if hasattr(env, 'reward_history') and len(env.reward_history) > 10:
        reward_std = np.std(env.reward_history[-10:])
        if reward_std < 0.01:
            diagnostics['issues'].append("Reward has very low variance (possibly no PnL component)")
    
    return diagnostics


def diagnose_training_stability(loss_history: List[float], episode: int) -> Dict[str, Any]:
    """
    Diagnose training stability to check for loss explosion.
    
    Args:
        loss_history: List of loss values
        episode: Current episode number
        
    Returns:
        Dictionary with diagnostic information
    """
    if len(loss_history) < 10:
        return {'issues': ['Not enough history']}
    
    recent_losses = loss_history[-10:]
    diagnostics = {
        'current_loss': float(recent_losses[-1]),
        'mean_loss': float(np.mean(recent_losses)),
        'loss_std': float(np.std(recent_losses)),
        'loss_trend': 'stable',
        'loss_exploding': False,
        'issues': []
    }
    
    # Check for loss explosion
    if len(loss_history) >= 20:
        early_losses = loss_history[-20:-10]
        late_losses = recent_losses
        if np.mean(late_losses) > 10 * np.mean(early_losses):
            diagnostics['loss_exploding'] = True
            diagnostics['loss_trend'] = 'exploding'
            diagnostics['issues'].append("Loss is exploding (10x+ increase)")
    
    # Check for NaN or Inf
    if np.any(np.isnan(recent_losses)) or np.any(np.isinf(recent_losses)):
        diagnostics['issues'].append("Loss contains NaN or Inf values")
    
    return diagnostics


def comprehensive_diagnosis(
    agent,
    env,
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    info: Dict[str, Any],
    loss_history: List[float],
    episode: int
) -> Dict[str, Any]:
    """
    Run comprehensive diagnosis of all components.
    
    Args:
        agent: Agent instance
        env: Environment instance
        obs: Current observation
        action: Action taken
        reward: Reward received
        info: Info from step
        loss_history: History of losses
        episode: Current episode
        
    Returns:
        Comprehensive diagnostic report
    """
    report = {
        'episode': episode,
        'action_diagnostics': diagnose_actions(action, {'low': -1, 'high': 1}),  # Adjust bounds
        'environment_diagnostics': diagnose_environment_state(env, info),
        'reward_diagnostics': diagnose_reward_function(env, action, reward),
        'stability_diagnostics': diagnose_training_stability(loss_history, episode),
        'summary': {
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
    }
    
    # Collect all issues
    all_issues = []
    all_issues.extend(report['action_diagnostics'].get('issues', []))
    all_issues.extend(report['environment_diagnostics'].get('issues', []))
    all_issues.extend(report['reward_diagnostics'].get('issues', []))
    all_issues.extend(report['stability_diagnostics'].get('issues', []))
    
    # Categorize issues
    critical_keywords = ['zero', 'not tracked', 'not available', 'exploding', 'NaN', 'Inf']
    warning_keywords = ['low', 'high', 'variance']
    
    for issue in all_issues:
        if any(keyword in issue.lower() for keyword in critical_keywords):
            report['summary']['critical_issues'].append(issue)
        elif any(keyword in issue.lower() for keyword in warning_keywords):
            report['summary']['warnings'].append(issue)
    
    # Generate recommendations
    if report['action_diagnostics'].get('zero_actions_ratio', 0) > 0.9:
        report['summary']['recommendations'].append(
            "Fix action outputs: Check action network, clipping, and normalization"
        )
    
    if report['stability_diagnostics'].get('loss_exploding', False):
        report['summary']['recommendations'].append(
            "Fix loss explosion: Add gradient clipping, reduce learning rate"
        )
    
    if not report['reward_diagnostics'].get('pnl_in_reward', False):
        report['summary']['recommendations'].append(
            "Add PnL to reward function: Include realized + unrealized PnL"
        )
    
    if report['environment_diagnostics'].get('orders_placed', 0) == 0:
        report['summary']['recommendations'].append(
            "Check order execution: Verify actions are being converted to orders"
        )
    
    return report


def print_diagnostic_report(report: Dict[str, Any]):
    """Print formatted diagnostic report."""
    print("\n" + "="*60)
    print(f"DIAGNOSTIC REPORT - Episode {report['episode']}")
    print("="*60)
    
    print("\n📊 ACTION DIAGNOSTICS:")
    action_diag = report['action_diagnostics']
    print(f"  Mean: {action_diag['action_mean']:.4f}")
    print(f"  Std: {action_diag['action_std']:.4f}")
    print(f"  Range: [{action_diag['action_min']:.4f}, {action_diag['action_max']:.4f}]")
    print(f"  Zero actions: {action_diag['zero_actions_ratio']:.2%}")
    if action_diag['issues']:
        print(f"  ⚠️  Issues: {', '.join(action_diag['issues'])}")
    
    print("\n🌍 ENVIRONMENT DIAGNOSTICS:")
    env_diag = report['environment_diagnostics']
    print(f"  Position: {env_diag['position']}")
    print(f"  Total PnL: {env_diag['total_pnl']}")
    print(f"  Realized PnL: {env_diag['realized_pnl']}")
    print(f"  Unrealized PnL: {env_diag['unrealized_pnl']}")
    print(f"  Orders placed: {env_diag['orders_placed']}")
    print(f"  Orders filled: {env_diag['orders_filled']}")
    if env_diag['issues']:
        print(f"  ⚠️  Issues: {', '.join(env_diag['issues'])}")
    
    print("\n🎯 REWARD DIAGNOSTICS:")
    reward_diag = report['reward_diagnostics']
    print(f"  Reward: {reward_diag['reward']:.4f}")
    print(f"  PnL in reward: {reward_diag['pnl_in_reward']}")
    if reward_diag['issues']:
        print(f"  ⚠️  Issues: {', '.join(reward_diag['issues'])}")
    
    print("\n📈 STABILITY DIAGNOSTICS:")
    stability_diag = report['stability_diagnostics']
    print(f"  Current loss: {stability_diag.get('current_loss', 'N/A')}")
    print(f"  Loss trend: {stability_diag.get('loss_trend', 'N/A')}")
    if stability_diag.get('loss_exploding', False):
        print(f"  ⚠️  Loss is exploding!")
    if stability_diag['issues']:
        print(f"  ⚠️  Issues: {', '.join(stability_diag['issues'])}")
    
    print("\n🔍 SUMMARY:")
    summary = report['summary']
    if summary['critical_issues']:
        print("  🚨 CRITICAL ISSUES:")
        for issue in summary['critical_issues']:
            print(f"    - {issue}")
    
    if summary['warnings']:
        print("  ⚠️  WARNINGS:")
        for warning in summary['warnings']:
            print(f"    - {warning}")
    
    if summary['recommendations']:
        print("  💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"    - {rec}")
    
    print("="*60 + "\n")


# Example usage in training loop:
"""
# In your training loop, add:
if episode % 10 == 0:
    # Get a sample action
    action = agent.select_action(obs)
    
    # Step environment
    next_obs, reward, done, info = env.step(action)
    
    # Run diagnosis
    report = comprehensive_diagnosis(
        agent=agent,
        env=env,
        obs=obs,
        action=action,
        reward=reward,
        info=info,
        loss_history=actor_loss_history,  # or critic_loss_history
        episode=episode
    )
    
    # Print report
    print_diagnostic_report(report)
"""

