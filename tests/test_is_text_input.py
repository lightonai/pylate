from __future__ import annotations

from pylate.models import ColBERT


class TestIsTextInput:
    """Test ColBERT._is_text_input heuristic for text vs multimodal detection."""

    def test_plain_string(self):
        assert ColBERT._is_text_input("hello") is True

    def test_list_of_strings(self):
        assert ColBERT._is_text_input(["hello", "world"]) is True

    def test_empty_list(self):
        assert ColBERT._is_text_input([]) is False

    def test_list_of_string_tuples(self):
        assert ColBERT._is_text_input([("hello", "world")]) is True

    def test_list_of_mixed_tuples(self):
        """Tuple with a non-string element should return False."""
        assert ColBERT._is_text_input([(42, "world")]) is False

    def test_dict_text_only(self):
        assert ColBERT._is_text_input([{"text": "hello"}]) is True

    def test_dict_with_image_key(self):
        assert ColBERT._is_text_input([{"image": "some_path"}]) is False

    def test_dict_with_images_key(self):
        assert ColBERT._is_text_input([{"images": "some_path"}]) is False

    def test_dict_with_audio_key(self):
        assert ColBERT._is_text_input([{"audio": "data"}]) is False

    def test_dict_with_video_key(self):
        assert ColBERT._is_text_input([{"video": "data"}]) is False

    def test_dict_with_pixel_values_key(self):
        assert ColBERT._is_text_input([{"pixel_values": "data"}]) is False

    def test_dict_with_non_string_values(self):
        """Dict with no multimodal keys but non-string values."""
        assert ColBERT._is_text_input([{"key": 42}]) is False

    def test_dict_mixed_text_and_image(self):
        """Dict with both text and image keys."""
        assert ColBERT._is_text_input([{"text": "hello", "image": "img"}]) is False

    def test_non_iterable(self):
        assert ColBERT._is_text_input(42) is False

    def test_none(self):
        assert ColBERT._is_text_input(None) is False
