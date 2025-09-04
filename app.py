#!/usr/bin/env python3
"""
执行脚本：分别运行s_true为True和False两种情况
"""
import subprocess
import sys
import os

def run_inference(s_true_value):
    """运行推理程序"""
    print(f"正在执行 s_true={s_true_value}")
    
    # 构建命令 - 正确传递bool参数
    if s_true_value:
        cmd = [sys.executable, 'inference.py', '--s_true', 'True']
    else:
        cmd = [sys.executable, 'inference.py', '--s_true', 'False']
    
    try:
        # 执行命令
        result = subprocess.run(cmd, 
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        print(f"s_true={s_true_value} 执行成功")
        print("输出:")
        print(result.stdout)
        
        if result.stderr:
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"s_true={s_true_value} 执行失败，返回码: {e.returncode}")
        print("输出:")
        print(e.stdout)
        print("错误信息:")
        print(e.stderr)
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
    
    print("-" * 50)

if __name__ == '__main__':
    print("开始执行两种模式的推理...")
    print("=" * 50)
    
    # 执行s_true=False
    run_inference(False)
    
    # 执行s_true=True  
    run_inference(True)
    
    print("所有模式执行完成！")
    print("结果保存在以下目录:")
    print("- s_true=False: log/ 目录")
    print("- s_true=True: s_log/ 目录")