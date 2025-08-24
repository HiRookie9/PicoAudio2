import os
import copy
import json
import time
import numpy as np
import torch
import argparse
import librosa
import soundfile as sf
from tqdm import tqdm 
import sys
from utils.grounding import groundtruth_inference

import sed_eval
import dcase_util

import laion_clap
from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
from sklearn.metrics.pairwise import cosine_similarity

def load_clap_scorer(ckpt_path):
    """
    Loads a CLAP scorer model with given checkpoint path, removes specified keys, and returns the model in eval mode.
    
    Args:
        ckpt_path (str): Path to the checkpoint file.
        
    Returns:
        clap_scorer: Loaded and ready CLAP scorer model.
    """
    clap_scorer = laion_clap.CLAP_Module(enable_fusion=False)
    ckpt = clap_load_state_dict(ckpt_path, skip_params=True)
    del_parameter_key = ["text_branch.embeddings.position_ids"]
    ckpt = {"model." + k: v for k, v in ckpt.items() if k not in del_parameter_key}
    clap_scorer.load_state_dict(ckpt)
    clap_scorer.eval()
    return clap_scorer

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

def get_event_list_and_count(onset_dict):
    """返回事件列表和每个事件的频次"""
    return list(onset_dict.keys()), {k: len(v) for k, v in onset_dict.items()}

def calculate_frequency_abs(gt_count, pred_count):
    abs_sum, n = 0, 0
    for event in gt_count:
        abs_sum += abs(gt_count.get(event, 0) - pred_count.get(event, 0))
        n += 1
    return abs_sum / n if n else 0

def calculate_sed_metric(ref_list, pred_list):

    reference_event_list = dcase_util.containers.MetaDataContainer(ref_list)
    estimated_event_list = dcase_util.containers.MetaDataContainer(pred_list)

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=2.0
    )
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=0.250
    )

    for filename in reference_event_list.unique_files:
        reference_event_list_for_current_file = reference_event_list.filter(
            filename=filename
        )

        estimated_event_list_for_current_file = estimated_event_list.filter(
            filename=filename
        )

        segment_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

        event_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    segment_f1 = segment_based_metrics.results_overall_metrics()['f_measure']['f_measure']
    return segment_f1

def main(gen_jsonl, ref_jsonl, caption_jsonl, onset_jsonl, result_txt):
    gen_data = {x['audio_id']: x['audio'] for x in read_jsonl(gen_jsonl)}
    ref_data = {x['audio_id']: x['audio'] for x in read_jsonl(ref_jsonl)}
    caption_data = {x['audio_id']: x['caption'] for x in read_jsonl(caption_jsonl)}
    onset_data = {x['audio_id']: x['onset'] for x in read_jsonl(onset_jsonl)}

    frequency_abs_list = []

    # 这里按dcase_eval要求，所有事件的dict全部压入list
    refs_list = []
    preds_list = []
    clap_out = []

    for audio_id in tqdm(onset_data, desc="Evaluating"):
        gt_onset_str = onset_data[audio_id]
        #print(gt_onset_str)
        gt_onset_dict = parse_onset(gt_onset_str)
        gt_event_list, gt_count = get_event_list_and_count(gt_onset_dict)

        gen_audio_path = gen_data[audio_id]
        # 用模型推理，得到生成音频的onset字符串
        pred_onset_str = groundtruth_inference(gen_audio_path, gt_event_list)
        #print(pred_onset_str)
        pred_onset_dict = parse_onset(pred_onset_str)
        #print(pred_onset_dict)
        _, pred_count = get_event_list_and_count(pred_onset_dict)
        #print(pred_count)
        # 统计频率绝对误差
        freq_abs = calculate_frequency_abs(gt_count, pred_count)
        frequency_abs_list.append(freq_abs)

        # 按dcase_eval要求格式：event_label, onset, offset, filename
        for event, spans in gt_onset_dict.items():
            #print(event)
            #print(spans)
            for onset, offset in spans:
                refs_list.append({
                    "event_label": event,
                    "onset": onset,
                    "offset": offset,
                    "filename": f"{audio_id}.wav"
                })
        for event, spans in pred_onset_dict.items():
            #print(event)
            #print(spans)
            for onset, offset in spans:
                preds_list.append({
                    "event_label": event,
                    "onset": onset,
                    "offset": offset,
                    "filename": f"{audio_id}.wav"
                })
                #clap score
        caption = caption_data[audio_id]
        # 2. 计算文本embedding
        text_embed = clap_scorer.get_text_embedding([caption, ""], use_tensor=False)[:1]
        # 3. 读取生成音频
        audio_data, _ = librosa.load(gen_audio_path, sr=48000)  # sample rate按你的模型需求
        audio_data = audio_data.reshape(1, -1)
        # 4. 计算音频embedding
        audio_embed = clap_scorer.get_audio_embedding_from_data(x=audio_data)
        # 5. 计算相似度
        pair_similarity = cosine_similarity(audio_embed, text_embed)[0][0]
        clap_out.append(pair_similarity)
        #break

    segment_f1 = calculate_sed_metric(refs_list, preds_list)
    frequency_abs_mean = np.mean(frequency_abs_list)
    print("frequency_abs:", frequency_abs_mean)
    print("segment_f1:", segment_f1)
    print("clap_score:", np.mean(clap_out))
    with open(result_txt, "a") as f:
        print("frequency_abs:", frequency_abs_mean, file=f)
        print("segment_f1:", segment_f1, file=f)
        print("clap_score:", np.mean(clap_out), file=f)

if __name__ == "__main__":
    # 这里用实际的文件路径替换
    main(
        gen_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_new/post_new/epoch43_7.5/gen_audio.jsonl",
        ref_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_new/post_new/epoch43_7.5/ref_audio.jsonl",
        caption_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_new/post_new/epoch43_7.5/ref_caption.jsonl",
        onset_jsonl="/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_new/post_new/epoch43_7.5/ref_onset.jsonl",
        result_txt = "/hpc_stor03/sjtu_home/zihao.zheng/data/audiocaps_v2/eval_json/pico_new/post_new/epoch43_7.5/result1.txt"
    )