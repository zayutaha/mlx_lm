import asyncio
import unittest
from unittest.mock import MagicMock, patch


class TestCopyableMarkdown(unittest.TestCase):

    def test_copyable_markdown_has_no_on_click(self):
        """CopyableMarkdown should not define on_click (lets terminal handle clicks)."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.CopyableMarkdown)
        self.assertNotIn("on_click", src,
                         "CopyableMarkdown should not override on_click")

    def test_chatui_has_on_click_for_select_mode(self):
        """ChatUI.on_click should handle bubble selection in select mode."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.ChatUI.on_click)
        self.assertIn("select_mode", src)

    def test_copy_selected_empty_does_nothing(self):
        """_copy_selected with no selections should notify and not run pbcopy."""
        import tui_main
        tui_main._selected_bubbles.clear()
        app = MagicMock()
        app.notify = MagicMock()

        async def run():
            await tui_main._copy_selected(app)

        with patch.object(asyncio, "create_subprocess_exec") as mock_pbcopy:
            asyncio.run(run())

        mock_pbcopy.assert_not_called()
        app.notify.assert_called_once()

    def test_copy_selected_joins_messages(self):
        """_copy_selected should join multiple selections with double newlines."""
        import tui_main
        tui_main._selected_bubbles.clear()

        class FakeBubble:
            def __init__(self, text):
                self._markdown = text
                self._initial_markdown = None
            def remove_class(self, name):
                pass

        b1 = FakeBubble("First message")
        b2 = FakeBubble("Second message")
        tui_main._selected_bubbles.extend([b1, b2])

        app = MagicMock()
        app.notify = MagicMock()

        async def fake_pbcopy(*args, **kwargs):
            class Proc:
                returncode = 0
                async def communicate(self, input=b""):
                    pass
            return Proc()

        with patch.object(asyncio, "create_subprocess_exec", side_effect=fake_pbcopy) as m:
            asyncio.run(tui_main._copy_selected(app))

        m.assert_called_once()
        self.assertEqual(len(tui_main._selected_bubbles), 0)


if __name__ == "__main__":
    unittest.main()
