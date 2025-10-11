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
    """
    Replace event names in both caption(TCC) and onset(TDC) string with corresponding free text descriptions.

    Args:
        caption (str): Caption text containing event names.
        onset (str): Onset string, formatted as "event__start-end--event2__start-end".

    Returns:
        new_caption (str): Caption with event names replaced by descriptions.
        new_onset (str): Onset string with event names replaced by descriptions.

    Notes:
        - Synonyms are fetched using get_event_synonyms().
        - For each event, a random synonym is chosen.
        - All occurrences in caption (with correct pluralization) and onset are replaced.
    """
    event_pattern = r"([a-zA-Z_()\s]+?)__((?:[\d\.\-]+_?)+)(?=--|$)"
    events = re.findall(event_pattern, onset)
    synonyms_dict = get_event_synonyms()
    replacements = {}
    # Choose a random synonym for each unique event
    for event_name, _ in events:
        if event_name not in replacements:
            candidates = synonyms_dict.get(event_name, [event_name])
            replacements[event_name] = random.choice(candidates) 
    # Replace event names in the onset string
    new_onset = "--".join([
        f"{replacements[event]}__{timestamps}"
        for event, timestamps in events
    ])
    # Replace event names in the caption, handling plural forms and case
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
    """
    Return replacement word with same plural suffix as the matched word.

    Args:
        match_obj (re.Match): Match object with possible plural suffix.
        replacement (str): Replacement string for the event name.

    Returns:
        str: Replacement string with plural suffix preserved.
    """

    matched = match_obj.group(0)
    suffix = match_obj.group(1) or ""  # Get plural suffix if present
    base_replacement = replacement

    # Preserve plural suffix ("s" or "es") from original word
    return base_replacement + suffix

if __name__ == "__main__":
    onset = "wind_chime__0.78-2.78"
    caption = "wind chime one times"

    print("Original onset:", onset)
    print("Original caption:", caption)
    
    caption, onset  = replace_event_synonyms(caption, onset)

    print("Modified onset:", onset)
    print("Modified caption:", caption)