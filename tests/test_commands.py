#!/usr/bin/env python3
"""
测试项目中的各个命令是否能正常运行
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path


class TestProjectCommands:
    """项目命令测试类"""

    @pytest.fixture
    def project_root(self):
        """获取项目根目录"""
        return Path(__file__).parent.parent

    def run_command(self, command, cwd=None, timeout=30):
        """运行命令并返回结果"""
        try:
            result = subprocess.run(
                command.split(),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "命令超时"
        except FileNotFoundError:
            return False, "", "命令不存在"
        except Exception as e:
            return False, "", f"未知错误: {e}"

    def test_project_directory_exists(self, project_root):
        """测试项目目录是否存在"""
        assert project_root.exists(), f"项目目录不存在: {project_root}"

    @pytest.mark.parametrize("file_path", [
        "examples/quick_start.py",
        "experiments/train_model.py",
        "experiments/evaluate_model.py",
        "configs/train_config.yaml",
        "configs/eval_config.yaml",
        "src/api/main.py"
    ])
    def test_required_files_exist(self, project_root, file_path):
        """测试必要文件是否存在"""
        full_path = project_root / file_path
        assert full_path.exists(), f"缺失文件: {file_path}"

    def test_module_import(self, project_root):
        """测试模块导入"""
        success, stdout, stderr = self.run_command(
            'python3 -c "import src.data.features; print(\'导入成功\')"',
            cwd=project_root
        )
        assert success, f"模块导入失败: {stderr}"
        assert "导入成功" in stdout

    def test_quick_start_example(self, project_root):
        """测试快速开始示例"""
        quick_start_path = project_root / "examples/quick_start.py"
        if quick_start_path.exists():
            success, stdout, stderr = self.run_command(
                "python3 examples/quick_start.py",
                cwd=project_root
            )
            assert success, f"快速开始示例运行失败: {stderr}"
        else:
            pytest.skip("quick_start.py文件不存在")