"""Shared fixtures. The tiny model is a randomly initialized Llama with a character-level
tokenizer built offline, so Part B code paths run on CPU without any download."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_outputs(tmp_path, monkeypatch):
    """Point every output path at a temp dir so tests never touch real logs/ledgers."""
    from advice_audit.common import config as C

    out = tmp_path / "outputs"
    monkeypatch.setattr(C, "OUTPUTS_DIR", out)
    monkeypatch.setattr(C, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(C, "PLANS_DIR", out / "plans")
    monkeypatch.setattr(C, "LOCAL_DIR", out / "local")
    import advice_audit.api.budget as B
    import advice_audit.api.runner as R
    import advice_audit.local.shards as S

    monkeypatch.setattr(B, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(R, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(S, "LOCAL_DIR", out / "local")
    import advice_audit.analysis.local_report as LR
    import advice_audit.analysis.report as RP
    import advice_audit.coding.coder as K
    import advice_audit.coding.human as H

    monkeypatch.setattr(K, "RUNS_DIR", out / "runs")
    monkeypatch.setattr(H, "HUMAN_DIR", out / "human_coding")
    monkeypatch.setattr(RP, "REPORTS_DIR", out / "reports")
    monkeypatch.setattr(LR, "LOCAL_DIR", out / "local")
    monkeypatch.setattr(LR, "OUTPUTS_DIR", out)
    yield out


CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _corpus() -> str:
    from advice_audit.materials.taxonomy import load_taxonomy

    parts = ["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?'\"()-/%&+\n",
             "我最推荐的是：我叫是女生男生女性男性。，、（）高考专业工作学生毕业大学省排名分语文数学外语物理化学生物历史政治地理"]
    for d in ("majors", "jobs"):
        t = load_taxonomy(d, with_attributes=False)
        parts += t["label_en"].tolist() + t["label_zh"].tolist()
    from advice_audit.common.config import load_cells, profile_ids
    from advice_audit.materials.render import render_prompt

    for d in ("majors", "jobs"):
        for cell in load_cells():
            for mode in ("open", "fixed"):
                for gc in ("f_explicit", "m_explicit"):
                    for i in range(3):
                        for para in range(3):
                            parts.append(render_prompt(d, mode, profile_ids(d)[0], cell, gc, i, para, strict=False).text)
    return "".join(parts)


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny_model")
    chars = sorted(set(_corpus()))
    specials = ["<pad>", "<s>", "</s>", "<unk>", "<|user|>", "<|assistant|>", "<|end|>"]
    vocab = {t: i for i, t in enumerate(specials + chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>", additional_special_tokens=specials[4:])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(d)
    return d


@pytest.fixture
def tiny(tiny_dir, monkeypatch):
    """(model, tokenizer, cfg) registered as local model key 'tiny'."""
    from advice_audit.common.config import LocalModelCfg, LocalModelsCfg, load_local_models
    import advice_audit.local.extract as E
    import advice_audit.local.modeling as M

    base = load_local_models()
    cfg = LocalModelCfg(key="tiny", repo_id=str(tiny_dir), revision="local", dtype="float32", role="dev")
    fake = LocalModelsCfg(models=[cfg], generation=base.generation, extraction={**base.extraction, "layers": "all"},
                          readout=base.readout)
    monkeypatch.setattr(M, "load_local_models", lambda: fake)
    monkeypatch.setattr(E, "load_local_models", lambda: fake)
    model = M.load_model(cfg, device="cpu")
    tok = M.load_tokenizer(cfg)
    return model, tok, cfg
