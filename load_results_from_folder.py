import os
import re
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font, Border, Side


# ================= 辅助函数 =================

def get_stats(values, confidence=0.95):
    """ 计算均值和标准差 """
    if not values:
        return np.nan, np.nan
    a = np.array(values)
    n = len(a)
    m = np.mean(a)
    if n < 2:
        return m, 0.0
    h = np.std(a, ddof=1)
    return m, h


def extract_metrics_from_line(line):
    """ 提取数值指标 (loss, acc等) """
    pattern = r"([a-zA-Z0-9_]+):(\d+(\.\d+)?)"
    matches = re.findall(pattern, line)
    metrics = {}
    for match in matches:
        key = match[0]
        value = float(match[1])
        metrics[key] = value
    return metrics


def extract_model_name_from_line(line):
    """
    【核心修改】针对你的日志格式提取模型名称
    匹配 "Model:" 后面的多个空格，然后抓取名称
    """
    # 你的日志里有 "Model ID:" 和 "Model:"
    # 我们只匹配 "Model:" (注意冒号紧跟在Model后面)，然后是 \s+ (任意空白字符，包括多个空格/Tab)
    pattern = r"Model:\s+([a-zA-Z0-9_\-\.]+)"

    match = re.search(pattern, line)
    if match:
        return match.group(1)  # 返回抓取到的名称
    return None


def remove_formatting(excel_path):
    """ 清除 Excel 样式 """
    try:
        wb = load_workbook(excel_path)
        ws = wb.active
        no_bold = Font(bold=False)
        no_border = Border(left=Side(style=None), right=Side(style=None),
                           top=Side(style=None), bottom=Side(style=None))
        for row in ws.iter_rows():
            for cell in row:
                cell.font = no_bold
                cell.border = no_border
        wb.save(excel_path)
    except Exception as e:
        print(f"清除样式忽略: {e}")


# ================= 主逻辑函数 =================

def process_logs(root_folder, output_excel):
    # 1. 获取基础 Batch Name
    batch_base_name = os.path.basename(os.path.normpath(root_folder))
    print(f"Batch Detected: {batch_base_name}")

    data_store = {}
    target_prefix = "[Stage2-Test]"

    # 用于存储扫描到的模型名称
    detected_model_name = None

    print(f"开始扫描文件夹: {root_folder} ...")

    for root, dirs, files in os.walk(root_folder):
        if "stage2.log" in files:
            dataset_name = os.path.basename(root)
            log_path = os.path.join(root, "stage2.log")

            if dataset_name not in data_store:
                data_store[dataset_name] = {}

            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # --- 修改点 1: 全局扫描模型名称 ---
                        # 无论这一行有没有 [Stage2-Test]，都要检查是不是配置行
                        if detected_model_name is None:
                            model_str = extract_model_name_from_line(line)
                            if model_str:
                                detected_model_name = model_str
                                print(f"  -> 成功提取模型名称: {detected_model_name}")

                        # --- 修改点 2: 提取测试数据 ---
                        # 只有包含 target_prefix 的行才包含 loss, acc 等数据
                        if target_prefix in line:
                            content = line.split(target_prefix)[1]
                            metrics = extract_metrics_from_line(content)
                            for k, v in metrics.items():
                                if k not in data_store[dataset_name]:
                                    data_store[dataset_name][k] = []
                                data_store[dataset_name][k].append(v)
            except Exception as e:
                print(f"读取出错: {log_path} -> {e}")

    if not data_store:
        print("未找到有效数据，跳过生成。")
        return

    # --- 3. 拼接最终索引名 (Batch + Model) ---
    if detected_model_name:
        final_row_index_name = f"{batch_base_name}_{detected_model_name}"
    else:
        final_row_index_name = batch_base_name
        print("警告: 未能在日志中找到 'Model:' 字段，将仅使用 Batch Name。")

    # --- 4. 准备数据 ---
    all_metrics = set()
    for ds in data_store.values():
        all_metrics.update(ds.keys())

    preferred_order = [
        "loss", "acc", "f1_macro", "recall_macro", "precision_macro",
        "infer_total_time", "infer_ms_per_sample", "infer_samples_per_sec", "infer_total_samples"
    ]
    remaining_metrics = sorted([m for m in all_metrics if m not in preferred_order])
    final_metric_order = [m for m in preferred_order if m in all_metrics] + remaining_metrics

    dataset_list = sorted(data_store.keys())

    row_data = {}
    for dataset_name in dataset_list:
        metrics_data = data_store[dataset_name]
        for metric in final_metric_order:
            values = metrics_data.get(metric, [])
            m, h = get_stats(values)
            row_data[(dataset_name, metric, "Mean")] = m
            row_data[(dataset_name, metric, "CI (95%)")] = h

    # --- 5. 生成 DataFrame ---
    # 使用拼接后的名字作为索引
    df = pd.DataFrame([row_data], index=[final_row_index_name])

    desired_columns = pd.MultiIndex.from_product(
        [dataset_list, final_metric_order, ["Mean", "CI (95%)"]],
        names=["Dataset", "Metric", "Stat"]
    )

    df = df.reindex(columns=desired_columns)
    df.index.name = "ID"  # 设置第一列左上角的名称

    # --- 6. 写入 Excel ---
    try:
        output_dir = os.path.dirname(output_excel)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_excel(output_excel)
        print(f"\n数据已写入: {output_excel}")
        print(f"生成的 ID 为: {final_row_index_name}")
        remove_formatting(output_excel)

    except Exception as e:
        print(f"保存 Excel 失败: {e}")


if __name__ == "__main__":
    # ================= 配置区域 =================




    ROOT_DIRECTORY = r"/root/autodl-tmp/SensorLLMLib_v2/run_log/batch_20260120_194435"
    OUTPUT_FILE = r"/root/autodl-tmp/SensorLLMLib_v2/run_log/batch_20260120_194435_experiment_results.xlsx"

    # ROOT_DIRECTORY_02 = r"/root/autodl-tmp/SensorLLMLib_v2/run_log/batch_20260117_210057"
    # OUTPUT_FILE_02 = r"/root/autodl-tmp/SensorLLMLib_v2/run_log/batch_20260117_210057_experiment_results.xlsx"

    # ===========================================

    if os.path.exists(ROOT_DIRECTORY):
        process_logs(ROOT_DIRECTORY, OUTPUT_FILE)
