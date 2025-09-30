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
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from functools import partial
from pathlib import Path

# SED/segment_f1 related
import sed_eval
import dcase_util

# CLAP
import laion_clap
from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
from sklearn.metrics.pairwise import cosine_similarity

# AudioLDM eval
from audioldm_eval import EvaluationHelper

# utils
from utils.general import read_jsonl_to_mapping
from utils.grounding import groundtruth_inference

def load_clap_scorer(ckpt_path):
    """Loads a CLAP scorer model with given checkpoint path."""
    clap_scorer = laion_clap.CLAP_Module(enable_fusion=False)
    ckpt = clap_load_state_dict(ckpt_path, skip_params=True)
    del_parameter_key = ["text_branch.embeddings.position_ids"]
    ckpt = {"model." + k: v for k, v in ckpt.items() if k not in del_parameter_key}
    clap_scorer.load_state_dict(ckpt)
    clap_scorer.eval()
    return clap_scorer

def parse_onset(onset_str):
    """Parse onset string to dict of events and their time spans."""
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
                except Exception:
                    continue
    return result  # {'event': [(onset, offset), ...], ...}

def get_event_list_and_count(onset_dict):
    """Return event list and event frequency dict."""
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
        time_resolution=0.04
    )
    #event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
    #    event_label_list=reference_event_list.unique_event_labels,
    #    t_collar=0.250
    #)

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
        #event_based_metrics.evaluate(
        #    reference_event_list=reference_event_list_for_current_file,
        #    estimated_event_list=estimated_event_list_for_current_file
        #)

    segment_f1 = segment_based_metrics.results_overall_metrics()['f_measure']['f_measure']
    return segment_f1

def get_common_folder_path(audio_dict):
    """
    Extract the common folder path from audio path dictionary.
    Returns common folder path and whether all files are in the same folder.
    """
    if not audio_dict:
        return None, False
    paths = list(audio_dict.values())
    parent_folders = [os.path.dirname(path) for path in paths]
    common_prefix = str(Path(os.path.commonpath(parent_folders)).resolve())
    is_same_folder = all(parent == parent_folders[0] for parent in parent_folders)
    return common_prefix, is_same_folder

def compute_clap_metrics(entry, args):
    audio_id, ref_caption, gen_audio = entry
    audio, _ = librosa.load(gen_audio, sr=48000)
    with torch.no_grad():
        text_embed = args.clap_scorer.get_text_embedding([ref_caption, ""], use_tensor=False)[:1]
        audio_embed = args.clap_scorer.get_audio_embedding_from_data(x=audio.reshape(1, -1), use_tensor=False)
        clap_sim = cosine_similarity(text_embed, audio_embed)
    return audio_id, {
        "CLAP_score": clap_sim,
    }

def evaluate(args):
    """Calculate FAD, FD, KL, CLAP, and optionally segment_f1 metrics."""
    ref_aid_to_audios = read_jsonl_to_mapping(args.ref_audio_jsonl, "audio_id", "audio")
    gen_aid_to_audios = read_jsonl_to_mapping(args.gen_audio_jsonl, "audio_id", "audio")
    ref_aid_to_captions = read_jsonl_to_mapping(args.ref_caption_jsonl, "audio_id", "caption")
    if args.f1_seg:
        ref_aid_to_onsets = read_jsonl_to_mapping(args.ref_onset_jsonl, "audio_id", "onset")

    # Evaluate FAD, FD, KL
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = EvaluationHelper(args.data_sr, args.device, backbone="cnn14")
    gen_folder_path, gen_is_same_folder = get_common_folder_path(gen_aid_to_audios)
    ref_folder_path, ref_is_same_folder = get_common_folder_path(ref_aid_to_audios)
    assert gen_is_same_folder == True, "Generated audio files must be in the same folder."
    assert ref_is_same_folder == True, "Reference audio files must be in the same folder."
    eval_result = evaluator.main(gen_folder_path, ref_folder_path)
    assert ref_aid_to_audios.keys() == gen_aid_to_audios.keys(), "Reference and generated audio IDs do not match"

    results = defaultdict(dict)
    results.update(eval_result)

    # CLAP calculation
    audio_ids = list(ref_aid_to_audios.keys())
    entries = [(aid, ref_aid_to_captions[aid], gen_aid_to_audios[aid]) for aid in audio_ids]
    clap_scorer = load_clap_scorer(args.ckpt_path)
    args.clap_scorer = clap_scorer

    # Multiprocessing CLAP scores
    with Pool(processes=args.num_workers) as pool:
        worker = partial(compute_clap_metrics, args=args)
        for audio_id, metrics in tqdm(pool.imap(worker, entries), total=len(entries), desc="Computing metrics"):
            for metric, value in metrics.items():
                results[metric][audio_id] = value

    # Optional SED/segment_f1 calculation
    if args.f1_seg:
        frequency_abs_list = []
        refs_list = []
        preds_list = []
        for audio_id in tqdm(audio_ids, desc="SED/segment_f1"):
            gt_onset_str = ref_aid_to_onsets[audio_id]
            gt_onset_dict = parse_onset(gt_onset_str)
            gt_event_list, gt_count = get_event_list_and_count(gt_onset_dict)

            gen_audio_path = gen_aid_to_audios[audio_id]
            pred_onset_str = groundtruth_inference(gen_audio_path, gt_event_list)
            pred_onset_dict = parse_onset(pred_onset_str)
            _, pred_count = get_event_list_and_count(pred_onset_dict)
            freq_abs = calculate_frequency_abs(gt_count, pred_count)
            frequency_abs_list.append(freq_abs)

            # Format for dcase_eval: event_label, onset, offset, filename
            for event, spans in gt_onset_dict.items():
                for onset, offset in spans:
                    refs_list.append({
                        "event_label": event,
                        "onset": onset,
                        "offset": offset,
                        "filename": f"{audio_id}.wav"
                    })
            for event, spans in pred_onset_dict.items():
                for onset, offset in spans:
                    preds_list.append({
                        "event_label": event,
                        "onset": onset,
                        "offset": offset,
                        "filename": f"{audio_id}.wav"
                    })

        segment_f1 = calculate_sed_metric(refs_list, preds_list)
        frequency_abs_mean = np.mean(frequency_abs_list)
        results['segment_f1'] = segment_f1
        results['frequency_abs'] = frequency_abs_mean

    # Output
    with open(args.output_file, "a") as writer:
        for metric, values in results.items():
            if metric == "CLAP_score":
                print_msg = f"{metric}: {np.mean([v[0][0] for v in values.values()]):.3f}"
                if args.clap_per_audio:
                    for audio_id, score in values.items():
                        score_msg = f"{audio_id}: {score[0][0]:.3f}"
                        print(score_msg, file=writer)
            elif metric == "segment_f1":
                print_msg = f"{metric}: {values:.3f}"
            elif metric == "frequency_abs":
                print_msg = f"{metric}: {values:.3f}"
            else:
                print_msg = f"{metric}: {values:.3f}" if isinstance(values, float) else f"{metric}: {values}"
            print(print_msg)
            print(print_msg, file=writer)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio_jsonl", "-r", type=str, required=True, help="path to reference audio jsonl file")
    parser.add_argument("--ref_caption_jsonl", "-rc", type=str, required=True, help="path to reference caption jsonl file")
    parser.add_argument("--gen_audio_jsonl", "-g", type=str, required=True, help="path to generated audio jsonl file")
    parser.add_argument("--output_file", "-o", type=str, required=True, help="path to output file")
    parser.add_argument("--num_workers", "-c", default=4, type=int, help="number of workers for parallel processing")
    parser.add_argument("--data_sr", type=int, default=16000, help="target sample rate")
    parser.add_argument("--clap_per_audio", "-p", action="store_true", help="calculate and store CLAP score for each audio clip")
    parser.add_argument("--ckpt_path", type=str, default="/hpc_stor03/sjtu_home/zihao.zheng/ldm/laion_clap/630k-audioset-best.pt", help="CLAP model checkpoint path")
    parser.add_argument("--ref_onset_jsonl", "-ro", type=str, default=None, help="path to reference onset jsonl file (for f1_seg)")
    parser.add_argument("--f1_seg", action="store_true", help="whether to calculate segment_f1 and frequency_abs (default: False)")
    args = parser.parse_args()
    evaluate(args)
