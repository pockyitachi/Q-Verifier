import re
import json


def extract_json_object(text, json_type="dict"):
    """
    Extracts a JSON object from a text string.

    Parameters:
    - text (str): The text containing the JSON data.
    - json_type (str): The type of JSON structure to look for ("dict" or "list").

    Returns:
    - dict or list: The extracted JSON object, or None if parsing fails.
    """
    try:
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'#.*', '', text)
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_pattern = r"({.*?})" if json_type == "dict" else r"(\[.*?\])"

    code_block_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def print_status(chat_history):
    print("*" * 100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>" * (len(chat[1]) - 1) + "\n")
    print("*" * 100)


def serialize_graph(graph):
    return {
        "nodes": [
            {
                "id": node,
                "attributes": graph.nodes[node]
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "attributes": graph.edges[u, v]
            }
            for u, v in graph.edges
        ]
    }