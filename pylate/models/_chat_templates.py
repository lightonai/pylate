"""Chat templates that reproduce ``colpali_engine``'s per-processor visual prompts.

Each registered template renders to exactly the string that the corresponding
``colpali_engine`` processor sends to ``processor(text=..., images=...)``. The
goal is bit-identical preprocessing when training/inference goes through
``processor.apply_chat_template`` instead of the colpali_engine processor.

Contract for every template:

- Input: a single conversation (list of messages). Only ``messages[0]`` is
  consulted; we never emit multi-turn formatting.
- ``messages[0].content`` is a list of typed dicts (``{type, text/image/video/...}``)
  because the ST input formatter always picks the ``"structured"`` message
  format for these VLMs.
- Image-bearing inputs render the ColPali ``visual_prompt_prefix``, substituting
  the user-supplied text item (when present) for the default ``"Describe the
  image."`` placeholder. Audio/video placeholders are not supported here
  (qwen_omni / video-capable backbones still need bespoke handling).
- Text-only inputs render the raw text with no chat wrapping, matching
  ``colpali_engine``'s ``process_texts`` which calls the processor directly on
  the raw query string. Query augmentation is appended later in
  ``ColBERT.preprocess``.

The keys are ``transformers`` ``config.model_type`` strings. When a backbone has
no entry, the caller is expected to fall back to the processor's existing
``chat_template``.
"""

from __future__ import annotations

# Shared template for the Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5 family.
# Matches ColQwen2/2.5/3/3.5 ``visual_prompt_prefix``:
#     "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
#     "Describe the image.<|im_end|><|endoftext|>"
_QWEN_VL_TEMPLATE = (
    "{%- set msg = messages[0] -%}"
    "{%- set ns = namespace(has_image=false, text='') -%}"
    "{%- for item in msg.content -%}"
    "{%- if item.type == 'image' -%}{%- set ns.has_image = true -%}"
    "{%- elif item.type == 'text' -%}{%- set ns.text = item.text -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if ns.has_image -%}"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{{ ns.text if ns.text else 'Describe the image.' }}"
    "<|im_end|><|endoftext|>"
    "{%- else -%}"
    "{{ ns.text }}"
    "{%- endif -%}"
)


# ColPali (PaliGemma). Matches ``visual_prompt_prefix = "<image><bos>Describe the image."``
# and the query path ``bos_token + query`` from ``ColPaliProcessor.process_texts``.
_PALIGEMMA_TEMPLATE = (
    "{%- set msg = messages[0] -%}"
    "{%- set ns = namespace(has_image=false, text='') -%}"
    "{%- for item in msg.content -%}"
    "{%- if item.type == 'image' -%}{%- set ns.has_image = true -%}"
    "{%- elif item.type == 'text' -%}{%- set ns.text = item.text -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if ns.has_image -%}"
    "<image><bos>{{ ns.text if ns.text else 'Describe the image.' }}"
    "{%- else -%}"
    "<bos>{{ ns.text }}"
    "{%- endif -%}"
)


# ColIdefics3. Matches
#     "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
_IDEFICS3_TEMPLATE = (
    "{%- set msg = messages[0] -%}"
    "{%- set ns = namespace(has_image=false, text='') -%}"
    "{%- for item in msg.content -%}"
    "{%- if item.type == 'image' -%}{%- set ns.has_image = true -%}"
    "{%- elif item.type == 'text' -%}{%- set ns.text = item.text -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if ns.has_image -%}"
    "<|im_start|>User:<image>"
    "{{ ns.text if ns.text else 'Describe the image.' }}"
    "<end_of_utterance>\nAssistant:"
    "{%- else -%}"
    "{{ ns.text }}"
    "{%- endif -%}"
)


COLPALI_CHAT_TEMPLATES: dict[str, str] = {
    # PaliGemma backbone — original ColPali.
    "paligemma": _PALIGEMMA_TEMPLATE,
    # Qwen2-VL family (Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5 all share the same prompt).
    "qwen2_vl": _QWEN_VL_TEMPLATE,
    "qwen2_5_vl": _QWEN_VL_TEMPLATE,
    "qwen3_vl": _QWEN_VL_TEMPLATE,
    "qwen3_vl_moe": _QWEN_VL_TEMPLATE,
    # Idefics3 backbone.
    "idefics3": _IDEFICS3_TEMPLATE,
    # ColModernVBert: same prompt structure as Idefics3 but with a <|begin_of_text|>
    # opener instead of <|im_start|>. Registered under the upstream model_type
    # (left as TODO until the model_type string is confirmed at load time).
}


# Key under which we register the ColPali-flavored template in
# ``processor.chat_template``. Matches the file name HF writes under
# ``additional_chat_templates/<name>.jinja`` on save.
COLPALI_TEMPLATE_NAME = "sentence_transformers"
