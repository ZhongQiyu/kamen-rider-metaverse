# evaluation.py

import numpy as np

# 示例任务相关性矩阵
relatedness_matrix = np.array([
    [1.0, 0.2, 0.4],
    [0.2, 1.0, 0.5],
    [0.4, 0.5, 1.0]
])

# 计算平均相关性评分
average_relatedness = relatedness_matrix.mean(axis=1)
print("Average Relatedness Scores:", average_relatedness)

from sklearn.metrics import mutual_info_score

# 示例任务数据
task1_data = [0, 1, 0, 1, 1, 0]
task2_data = [1, 0, 1, 0, 1, 1]

# 计算互信息
mi_score = mutual_info_score(task1_data, task2_data)
print("Mutual Information Score:", mi_score)

# 示例任务复杂性
task_complexities = [5, 15, 25]

# 设定复杂性阈值
complexity_threshold = 10
tasks_to_split = [i for i, complexity in enumerate(task_complexities) if complexity > complexity_threshold]

print("Tasks to Split:", tasks_to_split)

import time

def example_task():
    time.sleep(2)  # 模拟任务执行时间

# 测量任务执行时间
start_time = time.time()
example_task()
end_time = time.time()

execution_time = end_time - start_time
print("Execution Time:", execution_time)

# 设定时间阈值
time_threshold = 1.0
if execution_time > time_threshold:
    print("Task should be split")

# 设定权重
weights = {
    'relatedness': 0.3,
    'information_gain': 0.3,
    'complexity': 0.2,
    'execution_time': 0.2
}

# 示例数据
relatedness_scores = [0.5, 0.3, 0.7]
information_gain_scores = [0.4, 0.5, 0.6]
complexity_scores = [0.6, 0.2, 0.9]
execution_time_scores = [0.7, 0.4, 0.8]

# 计算综合评分
comprehensive_scores = [
    weights['relatedness'] * relatedness_scores[i] +
    weights['information_gain'] * information_gain_scores[i] +
    weights['complexity'] * complexity_scores[i] +
    weights['execution_time'] * execution_time_scores[i]
    for i in range(len(relatedness_scores))
]

print("Comprehensive Scores:", comprehensive_scores)

# 设定综合评分阈值
comprehensive_threshold = 0.6
tasks_to_split = [i for i, score in enumerate(comprehensive_scores) if score > comprehensive_threshold]

print("Tasks to Split:", tasks_to_split)

