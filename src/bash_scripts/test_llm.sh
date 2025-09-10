accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml llm_inference.py \
    input_text="a dog barks then a cat meows for two times" \
    input_onset=null \
    input_length="10.0" \
    time_control=False