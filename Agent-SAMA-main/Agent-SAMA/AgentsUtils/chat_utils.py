import base64
import requests
import json
import copy
import time


def image_encoder(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def match_model_name(model_name, candidates):
    return next((key for key in candidates if key in model_name.lower()), None)


def usage_tracker(res_json):
    """
    Extracts usage and cost information from a model response.
    """
    model = res_json.get("model")
    usage = res_json.get("usage", {})

    # Attempt to retrieve token counts from various possible key names
    prompt_tokens = usage.get("prompt_tokens") or usage.get("promptTokens") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("completionTokens") or usage.get("output_tokens")

    # Define per-million token pricing for different models
    pricing_per_million = {
        "gpt-4o": {"prompt": 2.5, "completion": 10},
        "gemini": {"prompt": 1.25, "completion": 5},
        "claude": {"prompt": 3, "completion": 15}
    }

    cost = {"prompt_token_price": None, "completion_token_price": None}
    if prompt_tokens is not None and completion_tokens is not None:
        matched_model = match_model_name(model, pricing_per_million)
        if matched_model:
            rate = pricing_per_million[matched_model]
            cost["prompt_token_price"] = (rate["prompt"] / 1_000_000) * prompt_tokens
            cost["completion_token_price"] = (rate["completion"] / 1_000_000) * completion_tokens

    return {
        "id": res_json.get("id"),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        **cost
    }


def build_headers(model, token, api_url):
    if "claude" in model and "47.88.8.18:8088" not in api_url:
        return {
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }


def build_message_payload(chat, model):
    data = {
        "messages": [],
    }

    if "claude" in model:
        for role, content in chat:
            if role == "system":
                assert content[0]['type'] == "text" and len(content) == 1
                data['system'] = content[0]['text']
            else:
                converted = []
                for item in content:
                    if item['type'] == "text":
                        converted.append({"type": "text", "text": item['text']})
                    elif item['type'] == "image_url":
                        converted.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": item['image_url']['url'].replace("data:image/jpeg;base64,", "")
                            }
                        })
                    else:
                        raise ValueError(f"Unsupported content type: {item['type']}")
                data['messages'].append({"role": role, "content": converted})
    else:
        for role, content in chat:
            data["messages"].append({"role": role, "content": content})

    return data


def send_request_with_retry(api_url, headers, data, is_claude, max_retry=5, sleep_sec=20):
    while max_retry > 0:
        try:
            res = requests.post(api_url, headers=headers,
                                data=json.dumps(data) if is_claude else None,
                                json=None if is_claude else data)
            res.raise_for_status()
            res_json = res.json()
            content = res_json['content'][0]['text'] if is_claude else res_json['choices'][0]['message']['content']
            return content, res_json
        except Exception as e:
            print("Request failed:", str(e))
            try:
                print(res.json())
            except:
                print("No valid JSON returned")
            max_retry -= 1
            if max_retry <= 0:
                print("Exceeded max retry attempts.")
                return None, None
            print(f"Retrying in {sleep_sec} seconds...")
            time.sleep(sleep_sec)


def inference_chat(chat, model, api_url, token, usage_tracking_jsonl=None, max_tokens=2048, temperature=0.0):
    if not token:
        raise ValueError("API key is required")

    headers = build_headers(model, token, api_url)
    data = build_message_payload(chat, model)
    
    # GPT-5.x uses max_completion_tokens instead of max_tokens
    if model.startswith("gpt-5"):
        data.update({
            "model": model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature
        })
    else:
        data.update({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature
        })

    is_claude = "claude" in model
    response_text, res_json = send_request_with_retry(api_url, headers, data, is_claude)

    if response_text and usage_tracking_jsonl and res_json:
        try:
            usage = usage_tracker(res_json)
            with open(usage_tracking_jsonl, "a") as f:
                f.write(json.dumps(usage) + "\n")
        except Exception as e:
            print(f"Failed to log usage: {e}")

    return response_text


def add_response_with_images(role, prompt, chat_history, images):
    """
    Adds a new message to the chat history with a text prompt and one or more images.

    Args:
        role (str): The role of the message sender, e.g., "user" or "assistant".
        prompt (str): The text content to include with the images.
        chat_history (list): Existing chat history (list of [role, content] pairs).
        images (list): A list of image objects (e.g., PIL images or paths) to be base64 encoded.

    Returns:
        list: A new chat history with the appended message.
    """
    new_chat_history = copy.deepcopy(chat_history)

    content = [{"type": "text", "text": prompt}]
    for image in images:
        base64_img = image_encoder(image)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    new_chat_history.append([role, content])
    return new_chat_history


def add_response(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)

    content = [{"type": "text", "text": prompt}]
    if image is not None:
        base64_image = image_encoder(image)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    new_chat_history.append([role, content])
    return new_chat_history

