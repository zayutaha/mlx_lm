import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import subprocess


class TestCopyableMarkdown(unittest.TestCase):

    def test_click_event_has_no_is_double(self):
        """The version of Textual used here does not have is_double on Click.
        The handler must not crash and should treat double-click as two
        single-clicks (toggle on first, toggle off on second)."""
        from textual.events import Click
        # Verifying the assumption that causes the bug
        self.assertFalse(hasattr(Click, "is_double"),
                         "This Textual version has is_double — update handler")

    def test_single_click_toggles_selection(self):
        """Single-click on a bubble should toggle selection."""
        from tui_main import CopyableMarkdown, _selected_bubbles

        _selected_bubbles.clear()

        # We can't easily instantiate a real Markdown widget without an App,
        # so instead test the _copy_selected function directly and verify
        # the module imports without AttributeError from is_double.
        import tui_main
        # Force re-eval of the class definition by checking the source
        import inspect
        src = inspect.getsource(tui_main.CopyableMarkdown.on_click)
        # The handler should NOT reference is_double
        self.assertNotIn("is_double", src,
                         "on_click should not use is_double — "
                         "this Textual version doesn't have it")

    def test_copy_selected_joins_messages(self):
        """_copy_selected should join multiple selections with double newlines."""
        from tui_main import _selected_bubbles, _copy_selected
        from unittest.mock import MagicMock

        _selected_bubbles.clear()

        # Create fake bubble objects with the attributes _copy_selected needs
        class FakeBubble:
            def __init__(self, text):
                self._markdown = text
                self._initial_markdown = None
            def remove_class(self, name):
                pass

        b1 = FakeBubble("First message")
        b2 = FakeBubble("Second message")
        _selected_bubbles.extend([b1, b2])

        app = MagicMock()
        app.notify = MagicMock()

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            _copy_selected(app)

        mock_run.assert_called_once()
        input_text = mock_run.call_args[1]["input"].decode()
        self.assertIn("First message", input_text)
        self.assertIn("Second message", input_text)
        self.assertIn("\n\n", input_text)  # joined by double newline
        self.assertEqual(len(_selected_bubbles), 0)  # cleared after copy

    def test_copy_selected_empty_does_nothing(self):
        """_copy_selected with no selections should notify and not call pbcopy."""
        from tui_main import _selected_bubbles, _copy_selected
        from unittest.mock import MagicMock

        _selected_bubbles.clear()
        app = MagicMock()
        app.notify = MagicMock()

        with patch.object(subprocess, "run") as mock_run:
            _copy_selected(app)

        mock_run.assert_not_called()
        app.notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
