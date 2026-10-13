import base64
import io
import json
from urllib.error import HTTPError

import pytest
from PIL import Image

from mobile_world.reflection.anthropic_reflector import AnthropicReflector, normalize_usage
from mobile_world.reflection.base import build_prompt
from mobile_world.runtime.utils.models import JSONAction


def make_reflector(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    return AnthropicReflector(tmp_path/'unused.env', tmp_path/'provider.jsonl', min_latency_ms=0)


def test_payload_keeps_shared_text_and_ordered_resized_images(tmp_path, monkeypatch):
    obj = make_reflector(tmp_path, monkeypatch)
    action = JSONAction(action_type="click", x=20, y=40)
    payload = obj.build_request(Image.new('RGB', (1080,2400), 'red'),
                                Image.new('RGB', (1080,2400), 'blue'), action, 'Open settings', 'Adjust brightness')
    blocks = payload['messages'][0]['content']
    assert blocks[0]['text'] == build_prompt('Adjust brightness', 'Open settings', action)
    assert [b['type'] for b in blocks] == ['text', 'image', 'image']
    for block, pixel in zip(blocks[1:], [(255,0,0), (0,0,255)]):
        img = Image.open(io.BytesIO(base64.b64decode(block['source']['data'])))
        assert img.size == (230,512) and img.getpixel((0,0)) == pixel
    assert payload['max_tokens'] == 256 and payload['temperature'] == 0
    assert 'thinking' not in payload and 'system' not in payload


def test_cached_input_includes_each_component_once():
    usage = normalize_usage(dict(input_tokens=100, output_tokens=12, cache_creation_input_tokens=40, cache_read_input_tokens=60))
    assert usage['prompt_tokens'] == 200 and usage['completion_tokens'] == 12
    assert usage['cached_tokens'] == 60


@pytest.mark.parametrize('raw', [None, {}, {'input_tokens':5}, {'input_tokens':-1,'output_tokens':3}])
def test_missing_or_invalid_usage_is_unknown(raw):
    assert normalize_usage(raw) is None


def test_actual_response_usage_and_independent_ledger(tmp_path, monkeypatch):
    obj = make_reflector(tmp_path, monkeypatch)
    response = dict(type='message',model=obj.model,id='test-response',stop_reason='end_turn',
                    content=[dict(type='text',text='No. The search query did not change.')],
                    usage=dict(input_tokens=300,output_tokens=12))
    monkeypatch.setattr('mobile_world.reflection.anthropic_reflector.urlopen',
                        lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode()))
    image = Image.new('RGB',(10,10))
    result = obj.verify(image,image,JSONAction(action_type='input_text',text='query'),'Search','Find item')
    assert not result.verdict and result.token_usage()['prompt_tokens']==300
    assert result.token_usage()['completion_tokens']==12
    events=[json.loads(line) for line in obj.ledger_path.read_text().splitlines()]
    assert [e['event'] for e in events] == ['request_started','response_received']
    assert events[1]['normalized_usage']==result.token_usage()
    assert 'fake-test-key' not in obj.ledger_path.read_text()


def test_error_is_counted_unknown_and_not_retried(tmp_path, monkeypatch):
    obj=make_reflector(tmp_path,monkeypatch)
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1)
        raise HTTPError(obj.endpoint,429,'rate limited',{},None)
    monkeypatch.setattr('mobile_world.reflection.anthropic_reflector.urlopen',fail)
    image=Image.new('RGB',(10,10))
    result=obj.verify(image,image,JSONAction(action_type='click',x=1,y=1),'Open','Open')
    assert result.verdict and result.token_usage() is None and calls==[1]
    events=[json.loads(line) for line in obj.ledger_path.read_text().splitlines()]
    assert events[-1]['error']=='HTTP 429'
