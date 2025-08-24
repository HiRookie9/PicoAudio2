import json
from tqdm import tqdm
from utils.grounding import groundtruth_inference

def parse_onset(onset_str):
    """解析onset字符串，返回事件列表和每个事件的起止时间段。"""
    result = {}
    events = onset_str.strip().split("--")
    for event in events:
        if not event:
            continue
        event_name, spans = event.split("__")
        result[event_name] = []
        for seg in spans.split("_"):
            if seg:
                try:
                    onset, offset = map(float, seg.split("-"))
                    result[event_name].append((onset, offset))
                except Exception as e:
                    continue
    return result  # {'a_man_speaks': [(start1, end1), ...], ...}

def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]

def calculate_frequency_abs_all(gt_onset_dict, pred_onset_dict):
    """
    统计所有事件 instance 的数量绝对误差之和及instance总数，返回整体平均
    """
    abs_sum = 0
    total_instances = 0
    all_events = set(gt_onset_dict.keys()) | set(pred_onset_dict.keys())
    for event in all_events:
        gt_count = len(gt_onset_dict.get(event, []))
        pred_count = len(pred_onset_dict.get(event, []))
        abs_sum += abs(gt_count - pred_count)
        total_instances += gt_count  # 以GT为准统计所有事件instance
    return abs_sum, total_instances

def main(gen_jsonl, onset_jsonl, result_txt):
    gen_data = {x['audio_id']: x['audio'] for x in read_jsonl(gen_jsonl)}
    onset_data = {x['audio_id']: x['onset'] for x in read_jsonl(onset_jsonl)}

    abs_sum = 0
    total_instances = 0

    for audio_id in tqdm(onset_data, desc="Evaluating"):
        gt_onset_str = onset_data[audio_id]
        gt_onset_dict = parse_onset(gt_onset_str)
        gen_audio_path = gen_data[audio_id]
        # 用模型推理，得到生成音频的onset字符串
        # groundtruth_inference 需由用户实现
        pred_onset_str = groundtruth_inference(gen_audio_path, list(gt_onset_dict.keys()), threshold=0.5, max_gap=0.4)
        pred_onset_dict = parse_onset(pred_onset_str)
        cur_abs_sum, cur_total_instances = calculate_frequency_abs_all(gt_onset_dict, pred_onset_dict)
        abs_sum += cur_abs_sum
        total_instances += cur_total_instances

    frequency_abs_mean = abs_sum / total_instances if total_instances > 0 else 0
    # 以add的形式写入
    with open(result_txt, "a") as f:
        f.write(f"frequency_abs_add: {frequency_abs_mean}\n")
    print("frequency_abs_add:", frequency_abs_mean)

if __name__ == "__main__":
    # 这里用实际的文件路径替换
    main(
        gen_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_fake/post/gen_audio.jsonl",
        #ref_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/tango2/post/frequency/ref_audio.jsonl",
        #caption_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/tango2/post/frequency/ref_caption.jsonl",
        onset_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_fake/post/ref_onset.jsonl",
        result_txt = "/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_fake/post/result1.txt"
    )