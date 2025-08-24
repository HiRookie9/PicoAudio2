import json

def get_event_synonyms():
    file_path = "./utils/merge_content.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    result = {}
    for item in data:
        event = item.get("event")
        phrases = item.get("phrases", [])
        result[event] = phrases
    
    return result


import random
import re

def replace_event_synonyms(caption, onset):
    event_pattern = r"([a-zA-Z_()\s]+?)__((?:[\d\.\-]+_?)+)(?=--|$)"
    events = re.findall(event_pattern, onset)
    synonyms_dict = get_event_synonyms()
    replacements = {}
    for event_name, _ in events:
        if event_name not in replacements:
            candidates = synonyms_dict.get(event_name, [event_name])
            replacements[event_name] = random.choice(candidates) 
    
    new_onset = "--".join([
        f"{replacements[event]}__{timestamps}"
        for event, timestamps in events
    ])
    
    new_caption = caption
    for orig, repl in replacements.items():
        orig_space = orig.replace("_", " ")
        repl_space = repl.replace("_", " ")

        escaped_orig_space = re.escape(orig_space)

        pattern = rf"(?<!\w){escaped_orig_space}(es|s)?(?!\w)"

        new_caption = re.sub(
            pattern,
            lambda m: match_plural(m, repl_space),
            new_caption,
            flags=re.IGNORECASE
        )
    
    return new_caption.capitalize(), new_onset

def match_plural(match_obj, replacement):

    matched = match_obj.group(0)
    suffix = match_obj.group(1) or ""  # 获取复数后缀（es 或 s）
    base_replacement = replacement

    # 保持复数后缀的一致性
    return base_replacement + suffix

if __name__ == "__main__":
    onset = "wind_chime__0.78-2.78"
    caption = "wind chime one times"

    print("Original onset:", onset)
    print("Original caption:", caption)
    
    caption, onset  = replace_event_synonyms(caption, onset)

    print("Modified onset:", onset)
    print("Modified caption:", caption)