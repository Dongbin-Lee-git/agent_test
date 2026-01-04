import json
import re

def clean_and_parse_json(text: str):
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match: text = match.group(1)
        else:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match: text = match.group(1)
        return json.loads(text)
    except:
        return None
