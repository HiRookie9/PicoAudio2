import os
import json
import numpy as np
import argparse
from tqdm import tqdm 
import dcase_util
import sed_eval
from utils.grounding import groundtruth_inference
def calculate_sed_metric(ref_list, pred_list, time_resolution=2.0):

    reference_event_list = dcase_util.containers.MetaDataContainer(ref_list)
    estimated_event_list = dcase_util.containers.MetaDataContainer(pred_list)

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=time_resolution
    )


    for filename in tqdm(reference_event_list.unique_files, desc="Evaluating files"):
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

    segment_f1 = segment_based_metrics.results_overall_metrics()['f_measure']['f_measure']
    return segment_f1

def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]

def parse_onset(onset_str):
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
    return result  # {'a_man_speaks': [(start1, end1), ...], ...}

def get_event_list_and_count(onset_dict):
    return list(onset_dict.keys()), {k: len(v) for k, v in onset_dict.items()}

def main(args):
    gen_data = {x['audio_id']: x['audio'] for x in read_jsonl(args.gen_jsonl)}
    onset_data = {x['audio_id']: x['onset'] for x in read_jsonl(args.onset_jsonl)}

    refs_list = []
    preds_list = []
    refs_list_multi = []
    preds_list_multi = []

    for audio_id in tqdm(onset_data, desc="Evaluating"):  
        gt_onset_str = onset_data[audio_id]
        gt_onset_dict = parse_onset(gt_onset_str)
        gt_event_list, gt_count = get_event_list_and_count(gt_onset_dict)

        gen_audio_path = gen_data[audio_id]
        pred_onset_str = groundtruth_inference(gen_audio_path, gt_event_list, 0.5, args.max_gap)
        pred_onset_dict = parse_onset(pred_onset_str)
        is_multi_event = len(gt_onset_dict) > 1
        for event, spans in gt_onset_dict.items():
            for onset, offset in spans:
                entry = {
                    "event_label": event,
                    "onset": onset,
                    "offset": offset,
                    "filename": f"{audio_id}.wav"
                }
                refs_list.append(entry)
                if is_multi_event:
                    refs_list_multi.append(entry)
        for event, spans in pred_onset_dict.items():
            for onset, offset in spans:
                entry = {
                    "event_label": event,
                    "onset": onset,
                    "offset": offset,
                    "filename": f"{audio_id}.wav"
                }
                preds_list.append(entry)
                if is_multi_event:
                    preds_list_multi.append(entry)               
    segment_f1 = calculate_sed_metric(refs_list, preds_list, time_resolution=args.time_resolution)
    segment_f1_multi = calculate_sed_metric(refs_list_multi, preds_list_multi, time_resolution=args.time_resolution)
    
    output_str = (
        f"max_gap: {args.max_gap}\n"
        f"time_resolution: {args.time_resolution}\n"
        f"segment_f1: {segment_f1}\n"
        f"segment_f1_multi: {segment_f1_multi}\n"
    )
    print(output_str)
    with open(args.output_file, "a") as f:
        f.write(output_str)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_jsonl",
        "-g",
        type=str,
        required=True,
        help="Path to generated audio jsonl file"
    )
    parser.add_argument(
        "--onset_jsonl",
        "-o",
        type=str,
        required=True,
        help="Path to onset jsonl file"
    )
    parser.add_argument(
        "--time_resolution",
        "-t",
        type=float,
        default=0.04,
        help="Time resolution for metric"
    )
    parser.add_argument(
        "--max_gap",
        "-m",
        type=float,
        default=1.0,
        help="Max gap for inference"
    )
    parser.add_argument(
        "--output_file",
        "-f",
        type=str,
        required=True,
        help="Path to output file"
    )
    args = parser.parse_args()

    main(args)