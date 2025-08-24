import torch
import argparse
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from functools import partial

import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm 

# Ref: https://github.com/haoheliu/audioldm_eval/tree/main
# The ref command for installing: pip install git+https://github.com/haoheliu/audioldm_eval
from audioldm_eval import EvaluationHelper

# Ref: https://github.com/LAION-AI/CLAP
# The ref command for installing: pip install laion-clap
import laion_clap
from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict

from utils.general import read_jsonl_to_mapping

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

def compute_clap_metrics(entry: tuple[str, str, str], args):
    audio_id, ref_caption, gen_audio = entry

    audio, _ = librosa.load(gen_audio, sr = 48000)         
    with torch.no_grad():
        text_embed = args.clap_scorer.get_text_embedding([ref_caption, ""], use_tensor=False)[:1]
        audio_embed = args.clap_scorer.get_audio_embedding_from_data(x = audio.reshape(1, -1), use_tensor=False)
        clap_sim = cosine_similarity(text_embed, audio_embed)

    return audio_id, {
        "CLAP_score": clap_sim,
    }

def get_common_folder_path(audio_dict):
    """
    Extract the common folder path from audio path dictionary.
    
    Parameters:
    audio_dict -- Dictionary in format {audio_id: audio_path}
    
    Returns:
    common_folder -- Common folder path (None if no common path)
    is_same_folder -- Boolean indicating if all audios are in the same folder
    """
    if not audio_dict:
        return None, False   
    paths = list(audio_dict.values())
    parent_folders = [os.path.dirname(path) for path in paths]
    common_prefix = str(Path(os.path.commonpath(parent_folders)).resolve())
    is_same_folder = all(parent == parent_folders[0] for parent in parent_folders)
    
    return common_prefix, is_same_folder

def evaluate(args):
    """Calculate FAD, FD, KL, etc. socres."""
    ref_aid_to_audios = read_jsonl_to_mapping(
        args.ref_audio_jsonl, "audio_id", "audio"
    )
    gen_aid_to_audios = read_jsonl_to_mapping(
        args.gen_audio_jsonl, "audio_id", "audio"
    )

    """Calculate ldm eval score: FAD, FD, KL score"""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = EvaluationHelper(args.data_sr, args.device, backbone="cnn14")
    gen_folder_path, gen_is_same_folder = get_common_folder_path(gen_aid_to_audios)
    ref_folder_path, ref_is_same_folder = get_common_folder_path(ref_aid_to_audios)
    assert gen_is_same_folder == True, "Generated audio files must be in the same folder."
    assert ref_is_same_folder == True, "Reference audio files must be in the same folder."
    eval_result = evaluator.main(gen_folder_path, ref_folder_path)
    assert ref_aid_to_audios.keys() == gen_aid_to_audios.keys(
    ), "Reference and generated audio IDs do not match"
 
    results = defaultdict(dict)
    results.update(eval_result)

    """The CLAP calculation still needs to be verified."""

    audio_ids = list(ref_aid_to_audios.keys())
    ref_aid_to_captions = read_jsonl_to_mapping(
        args.ref_caption_jsonl, "audio_id", "caption"
    )
    entries = [(aid, ref_aid_to_captions[aid], gen_aid_to_audios[aid])
               for aid in audio_ids]
    
    clap_scorer = load_clap_scorer(args.ckpt_path)
    args.clap_scorer = clap_scorer
    # Parallel
    with Pool(processes=args.num_workers) as pool:
        worker = partial(compute_clap_metrics, args=args)
        for audio_id, metrics in tqdm(
            pool.imap(worker, entries),
            total=len(entries),
            desc="Computing metrics"
        ):
            for metric, value in metrics.items():
                results[metric][audio_id] = value


    with open(args.output_file, "a") as writer:
        for metric, values in results.items():
            if metric == "CLAP_score":
                print_msg = f"{metric}: {np.mean(list(values.values())):.3f}"
                if args.clap_per_audio:
                    for audio_id, score in values.items():
                        score_msg = f"{audio_id}: {score[0][0]:.3f}"
                        print(score_msg, file=writer)
            else:
                print_msg = f"{metric}: {values:.3f}"
            print(print_msg)
            print(print_msg, file=writer)
    

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True) 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_audio_jsonl",
        "-r",
        type=str,
        required=True,
        help="path to reference audio jsonl file"
    )
    parser.add_argument(
        "--ref_caption_jsonl",
        "-rc",
        type=str,
        required=True,
        help="path to reference caption jsonl file"
    )
    parser.add_argument(
        "--gen_audio_jsonl",
        "-g",
        type=str,
        required=True,
        help="path to generated audio jsonl file"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        required=True,
        help="path to output file"
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        default=4,
        type=int,
        help="number of workers for parallel processing"
    )
    parser.add_argument(
        "--data_sr", 
        type=int, 
        default=16000,
        help="target sample rate"
    )
    parser.add_argument(
        "--clap_per_audio",
        "-p",
        action="store_true",
        help="calculate and store CLAP score for each audio clip"
    )
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/hpc_stor03/sjtu_home/zihao.zheng/ldm/laion_clap/630k-audioset-best.pt",
        help="target sample rate"
    )
    
    args = parser.parse_args()

    evaluate(args)
