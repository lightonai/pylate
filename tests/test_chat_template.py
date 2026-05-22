"""Smoke tests for :meth:`ColBERT._configure_chat_template`.

These exercise the resolution priority and the save/load round-trip without
loading a real VLM (which is expensive). We construct a minimal stand-in for the
loaded ``Transformer`` module and processor and call ``_configure_chat_template``
directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pylate.models._chat_templates import (
    COLPALI_CHAT_TEMPLATES,
    COLPALI_TEMPLATE_NAME,
)
from pylate.models.colbert import ColBERT


def _make_module(model_type: str | None, existing_chat_template) -> SimpleNamespace:
    """Build the minimal duck-typed surface ``_configure_chat_template`` reads."""
    processor = SimpleNamespace(chat_template=existing_chat_template)
    config = SimpleNamespace(model_type=model_type) if model_type else None
    model = SimpleNamespace(config=config) if config else None
    return SimpleNamespace(
        processor=processor,
        model=model,
        processing_kwargs={},
    )


def _configure(module, user_processor_kwargs=None):
    """Drive ``_configure_chat_template`` with a stand-in ``first_module``.

    We bypass ``ColBERT.__init__`` and just patch ``_first_module`` so the method
    sees our fake module.
    """
    colbert = ColBERT.__new__(ColBERT)
    colbert._first_module = lambda: module  # type: ignore[method-assign]
    ColBERT._configure_chat_template(colbert, user_processor_kwargs=user_processor_kwargs)


def test_registry_default_installed_for_qwen2_5_vl():
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template="old qwen tmpl")
    _configure(module)

    assert isinstance(module.processor.chat_template, dict)
    assert module.processor.chat_template["default"] == "old qwen tmpl"
    assert (
        module.processor.chat_template[COLPALI_TEMPLATE_NAME]
        == COLPALI_CHAT_TEMPLATES["qwen2_5_vl"]
    )
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )


def test_unknown_model_type_leaves_template_alone():
    module = _make_module(model_type="bert", existing_chat_template=None)
    _configure(module)

    assert module.processor.chat_template is None
    assert "chat_template" not in module.processing_kwargs


def test_user_override_is_installed_into_slot_for_persistence():
    # When the user pins a chat_template at construction, we install their
    # value under the sentence_transformers slot so save_pretrained writes it
    # to disk (additional_chat_templates/sentence_transformers.jinja), and we
    # rewire the ST kwarg to resolve through the slot.
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template="orig")

    _configure(
        module,
        user_processor_kwargs={"chat_template": {"chat_template": "{{ messages[0] }}"}},
    )

    # The user's template lives in the slot; the model's original is preserved.
    assert isinstance(module.processor.chat_template, dict)
    assert module.processor.chat_template[COLPALI_TEMPLATE_NAME] == "{{ messages[0] }}"
    assert module.processor.chat_template["default"] == "orig"
    # The kwarg is rewired to the slot name so apply_chat_template resolves to it.
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )
    # Registry must NOT win over the user's pin even when model_type matches.
    assert (
        module.processor.chat_template[COLPALI_TEMPLATE_NAME]
        != COLPALI_CHAT_TEMPLATES["qwen2_5_vl"]
    )


def test_user_passing_slot_name_falls_through():
    # If the user passes chat_template="sentence_transformers" they're saying
    # "use whatever the slot holds" — installing the literal name as content
    # would be a broken self-referential template. Fall through to cases (2)–
    # (4) so the registry (or persisted state) fills the slot.
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template="orig")

    _configure(
        module,
        user_processor_kwargs={
            "chat_template": {"chat_template": COLPALI_TEMPLATE_NAME}
        },
    )

    # Registry fired because case (1) deferred.
    assert isinstance(module.processor.chat_template, dict)
    assert (
        module.processor.chat_template[COLPALI_TEMPLATE_NAME]
        == COLPALI_CHAT_TEMPLATES["qwen2_5_vl"]
    )
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )


def test_user_override_does_not_mutate_caller_dict():
    # _set_chat_template_name must NOT mutate the inner dict the user passed —
    # ST stores processing_kwargs by reference, so an in-place mutation would
    # surprise callers holding a reference to their original dict.
    user_inner = {"chat_template": "user-jinja", "add_generation_prompt": False}
    user_outer = {"chat_template": user_inner}
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template="orig")

    _configure(module, user_processor_kwargs=user_outer)

    # The user's outer dict and inner dict are unchanged.
    assert user_outer == {"chat_template": user_inner}
    assert user_inner == {"chat_template": "user-jinja", "add_generation_prompt": False}


def test_persisted_colpali_pin_is_respected():
    # When sentence_bert_config.json round-trips our named pin, the loaded
    # module comes back with processing_kwargs["chat_template"]["chat_template"]
    # == "sentence_transformers". That's our checkpoint signature; leave it.
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template="orig-not-a-dict")
    module.processing_kwargs["chat_template"] = {
        "chat_template": COLPALI_TEMPLATE_NAME
    }

    _configure(module, user_processor_kwargs=None)

    # Registry must NOT touch the processor; persisted wiring stays as-is.
    assert module.processor.chat_template == "orig-not-a-dict"
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )


def test_persisted_non_colpali_pin_falls_through_to_registry():
    # If the persisted kwarg is anything *other* than our named pin (e.g. a raw
    # Jinja string the user previously set, or a different named template), we
    # treat it as "not a ColPali pin" and the registry default wins. This lets
    # users load a generic VLM checkpoint and still get ColPali preprocessing
    # without manual intervention.
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template="orig")
    module.processing_kwargs["chat_template"] = {"chat_template": "some-other-tmpl"}

    _configure(module, user_processor_kwargs=None)

    # Registry installed.
    assert isinstance(module.processor.chat_template, dict)
    assert (
        module.processor.chat_template[COLPALI_TEMPLATE_NAME]
        == COLPALI_CHAT_TEMPLATES["qwen2_5_vl"]
    )
    # And the named pin is overwritten with ours.
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )


def test_unrelated_chat_template_kwargs_dont_block_registry():
    # processing_kwargs["chat_template"] might carry other apply_chat_template
    # kwargs (e.g. add_generation_prompt=False) without a "chat_template" key.
    # That's not a pin — registry must still install.
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template="orig")
    module.processing_kwargs["chat_template"] = {"add_generation_prompt": False}

    _configure(module, user_processor_kwargs=None)

    assert isinstance(module.processor.chat_template, dict)
    assert COLPALI_TEMPLATE_NAME in module.processor.chat_template
    # The unrelated kwarg survives.
    assert module.processing_kwargs["chat_template"]["add_generation_prompt"] is False
    # And our named pin is added alongside it.
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )


def test_persisted_hf_template_is_respected():
    # If the checkpoint shipped `additional_chat_templates/sentence_transformers.jinja`
    # (HF processor mechanism) but the ST-side wiring wasn't persisted, we still
    # honor the HF half — and re-wire the kwarg so apply_chat_template uses it.
    saved = "{# previously-saved colpali template via HF processor #}"
    existing = {"default": "model default", COLPALI_TEMPLATE_NAME: saved}
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template=existing)

    _configure(module, user_processor_kwargs=None)

    assert module.processor.chat_template is existing
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )


def test_saved_sentence_transformers_template_is_reused():
    # Simulate a checkpoint that already shipped additional_chat_templates/sentence_transformers.jinja:
    # HF's `from_pretrained` would have populated `processor.chat_template` as a
    # dict containing our key.
    saved = "{# pretend this is a previously-saved colpali template #}"
    existing = {"default": "model default", COLPALI_TEMPLATE_NAME: saved}
    module = _make_module(model_type="qwen2_5_vl", existing_chat_template=existing)

    _configure(module)

    # The saved template wins over the registry default — checkpoint is source of truth.
    assert module.processor.chat_template is existing
    assert module.processor.chat_template[COLPALI_TEMPLATE_NAME] == saved
    assert (
        module.processing_kwargs["chat_template"]["chat_template"]
        == COLPALI_TEMPLATE_NAME
    )


def test_no_processor_is_noop():
    module = SimpleNamespace(processor=None)

    colbert = ColBERT.__new__(ColBERT)
    colbert._first_module = lambda: module  # type: ignore[method-assign]
    # Should not raise.
    ColBERT._configure_chat_template(colbert, user_processor_kwargs=None)


def test_save_load_round_trip_writes_named_jinja_file(tmp_path):
    """Verify the dict shape we install actually persists as
    ``additional_chat_templates/sentence_transformers.jinja`` and reloads.

    Uses a tokenizer (not a full multimodal processor) because tokenizers
    share the same save/load contract for ``chat_template`` dicts and the tiny
    Qwen2 test fixture is cheap to load offline. ``ProcessorMixin.save_pretrained``
    runs the same code path on the same ``chat_template`` attribute.

    Skipped when the fixture isn't in the local HF cache (e.g. CI without it).
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
        )
    except Exception as exc:
        pytest.skip(f"tiny qwen2 fixture unavailable: {exc}")

    template = COLPALI_CHAT_TEMPLATES["qwen2_5_vl"]
    tokenizer.chat_template = {
        "default": tokenizer.chat_template,
        COLPALI_TEMPLATE_NAME: template,
    }
    tokenizer.save_pretrained(tmp_path)

    addl = tmp_path / "additional_chat_templates" / f"{COLPALI_TEMPLATE_NAME}.jinja"
    assert addl.is_file(), "named template was not written to additional_chat_templates/"
    assert addl.read_text() == template
    assert (tmp_path / "chat_template.jinja").is_file()  # default template

    # Reload via HF and confirm the dict round-trips.
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert isinstance(reloaded.chat_template, dict)
    assert reloaded.chat_template[COLPALI_TEMPLATE_NAME] == template
    assert "default" in reloaded.chat_template


@pytest.mark.parametrize("model_type", sorted(COLPALI_CHAT_TEMPLATES))
def test_every_registered_template_renders_without_error(model_type):
    """Every registered Jinja template must compile and render both modalities."""
    import jinja2

    template_src = COLPALI_CHAT_TEMPLATES[model_type]
    template = jinja2.Environment().from_string(template_src)

    image_msg = [{"role": "user", "content": [{"type": "image", "image": object()}]}]
    text_msg = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    image_text_msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": object()},
                {"type": "text", "text": "custom doc prompt"},
            ],
        }
    ]

    image_render = template.render(messages=image_msg)
    text_render = template.render(messages=text_msg)
    both_render = template.render(messages=image_text_msg)

    # Image branch must include the model's image placeholder token.
    image_tokens = {
        "paligemma": "<image>",
        "qwen2_vl": "<|image_pad|>",
        "qwen2_5_vl": "<|image_pad|>",
        "qwen3_vl": "<|image_pad|>",
        "qwen3_vl_moe": "<|image_pad|>",
        "idefics3": "<image>",
    }
    assert image_tokens[model_type] in image_render
    # Default caption appears when no text item is provided.
    assert "Describe the image." in image_render
    # Provided text overrides the default caption.
    assert "custom doc prompt" in both_render
    assert "Describe the image." not in both_render
    # Text-only renders the raw query (PaliGemma prepends <bos>, others emit raw).
    assert "hello" in text_render
