"""Part B on a tiny random Llama (CPU): hooks, readout, steering, patching, extraction."""
import numpy as np
import pandas as pd
import pytest
import torch

from advice_audit.common.config import load_experiment
from advice_audit.local.hooks import add_vector, capture, decoder_layers, n_layers, patch
from advice_audit.local.modeling import encode_with_positions, format_chat
from advice_audit.local.readout import label_logprobs, label_texts, label_token_ids
from advice_audit.materials.design import expand_design


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def test_layer_resolution(tiny):
    model, _, _ = tiny
    assert len(decoder_layers(model)) == 3 == n_layers(model)


def test_capture_matches_output_hidden_states(tiny):
    model, tok, _ = tiny
    ids = torch.tensor([_ids(tok, "<|user|>Which majors should I consider?<|end|><|assistant|>")])
    with torch.no_grad(), capture(model, range(n_layers(model)), store_all=True) as store:
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    for l in range(n_layers(model) - 1):  # last hidden state has the final norm applied
        torch.testing.assert_close(store[l], hs[l + 1], atol=1e-5, rtol=1e-4)


def test_capture_positions_subset(tiny):
    model, tok, _ = tiny
    ids = torch.tensor([_ids(tok, "abcdefgh"), _ids(tok, "hgfedcba")])
    pos = torch.tensor([2, 5])
    with torch.no_grad(), capture(model, [1], positions=pos) as sel, capture(model, [1], store_all=True) as full:
        model(input_ids=ids)
    torch.testing.assert_close(sel[1][0], full[1][0, 2])
    torch.testing.assert_close(sel[1][1], full[1][1, 5])


def test_readout_cache_equals_full_sequence(tiny):
    model, tok, cfg = tiny
    chat = format_chat(tok, "Which majors should I consider?", cfg, "My top recommendation is:")
    p = _ids(tok, chat.text)
    _, labels = label_texts("majors", "en")
    lab = label_token_ids(tok, labels[:8])
    a, _ = label_logprobs(model, p, lab, use_cache=True)
    b, _ = label_logprobs(model, p, lab, use_cache=False)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_readout_invariant_to_batch_composition(tiny):
    """Label log-likelihoods do not depend on which other labels share the batch (padding)."""
    model, tok, cfg = tiny
    p = _ids(tok, format_chat(tok, "我应该考虑哪些专业？", cfg, "我最推荐的是：").text)
    _, labels = label_texts("majors", "zh")
    lab = label_token_ids(tok, labels)
    full, _ = label_logprobs(model, p, lab)
    for use_cache in (True, False):
        sub, _ = label_logprobs(model, p, lab[3:5], use_cache=use_cache)
        np.testing.assert_allclose(sub, full[3:5], atol=1e-4)
        rev, _ = label_logprobs(model, p, lab[::-1][:10], use_cache=use_cache)
        np.testing.assert_allclose(rev, full[::-1][:10], atol=1e-4)


def _greedy(model, tok, text, n=12):
    ids = torch.tensor([_ids(tok, text)])
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return out[0].tolist()


def test_steering_alpha_zero_reproduces_greedy(tiny):
    model, tok, cfg = tiny
    text = format_chat(tok, "Which jobs should I consider?", cfg).text
    ref = _greedy(model, tok, text)
    d = model.config.hidden_size
    v = torch.randn(d)
    with add_vector(model, {1: v * 0.0}):
        assert _greedy(model, tok, text) == ref
    with add_vector(model, {1: v * 50.0}):
        assert _greedy(model, tok, text) != ref  # sanity: a large vector changes output
    assert _greedy(model, tok, text) == ref  # hooks removed


def test_patch_sets_values_and_identity_patch_is_noop(tiny):
    model, tok, cfg = tiny
    ids = torch.tensor([_ids(tok, "abcdefghij")])
    with torch.no_grad(), capture(model, [0, 1], store_all=True) as clean:
        base = model(input_ids=ids).logits
    # patch layer 0 position 4 with its own value -> identical logits
    with torch.no_grad(), patch(model, 0, torch.tensor([4]), clean[0][:, 4]):
        same = model(input_ids=ids).logits
    torch.testing.assert_close(base, same)
    newval = torch.ones(1, model.config.hidden_size)
    with torch.no_grad(), patch(model, 0, torch.tensor([4]), newval), capture(model, [0], store_all=True) as st:
        changed = model(input_ids=ids).logits
    torch.testing.assert_close(st[0][0, 4], newval[0])
    assert not torch.allclose(base, changed)


def test_token_positions_for_name_and_cue(tiny):
    model, tok, cfg = tiny
    exp = load_experiment("pilot")
    exp.cells, exp.samples, exp.modes, exp.domains = ["A", "C"], 1, ["open"], ["jobs"]
    d = expand_design(exp, strict=False)
    import json

    for _, r in d[d["gender_cond"] == "f_explicit"].iterrows():
        chat = format_chat(tok, r["prompt"], cfg, "x")
        spans = {"name_last": tuple(json.loads(r["name_span"])), "cue_last": tuple(json.loads(r["cue_span"]))}
        ids, pos = encode_with_positions(tok, chat, spans)
        name_last_char = r["name"][-1]
        assert tok.decode([ids[pos["name_last"]]]) == name_last_char
        assert pos["last"] == len(ids) - 1


def test_extract_rows_end_to_end(tiny):
    from advice_audit.local.extract import extract_rows

    model, tok, cfg = tiny
    exp = load_experiment("pilot")
    exp.cells, exp.samples, exp.modes, exp.domains = ["A"], 1, ["open"], ["majors"]
    d = expand_design(exp, strict=False).head(10)
    acts, valid, pos = extract_rows(model, tok, cfg, d, ["name_last", "cue_last", "last"], [0, 1, 2], batch_size=4)
    assert acts.shape == (10, 3, 3, model.config.hidden_size)
    base = (d["gender_cond"] == "baseline").to_numpy()
    assert not valid[torch.as_tensor(base), 0].any()           # no name in baseline
    assert valid[:, 2].all()                                   # last token always present
    # batch independence: same rows extracted one by one
    acts1, _, _ = extract_rows(model, tok, cfg, d.iloc[[3]], ["name_last", "cue_last", "last"], [0, 1, 2], batch_size=1)
    torch.testing.assert_close(acts1[0, 2], acts[3, 2], atol=2e-3, rtol=1e-2)


def test_runtime_readout_summary(tiny, monkeypatch):
    from advice_audit.local.runtime import Runtime

    model, tok, cfg = tiny
    rt = Runtime("tiny", model=model, tok=tok)
    exp = load_experiment("pilot")
    exp.cells, exp.samples, exp.modes, exp.domains = ["A"], 1, ["fixed"], ["jobs"]
    row = expand_design(exp, strict=False).iloc[0]
    s, lp = rt.readout(row)
    assert set(["E_female_share_us", "E_prestige_isei", "entropy", "top_category"]) <= set(s)
    assert np.isfinite(lp).all()
