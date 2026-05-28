import asyncio
import unittest
from unittest.mock import MagicMock, patch


class TestCopyableMarkdown(unittest.TestCase):

    def test_click_has_ctrl_attr(self):
        """Verify the Click event has the 'ctrl' attribute we depend on."""
        from textual.events import Click
        self.assertTrue(hasattr(Click, "ctrl"),
                        "Click event missing 'ctrl' attribute")
        self.assertFalse(hasattr(Click, "is_ctrl"),
                         "Click has 'is_ctrl' — fix handler to use it")
        self.assertFalse(hasattr(Click, "is_double"),
                         "Click has 'is_double' — fix handler to use it")

    def test_on_click_uses_ctrl_not_is_ctrl(self):
        """The on_click handler should reference event.ctrl, not event.is_ctrl."""
        import tui_main
        import inspect
        src = inspect.getsource(tui_main.CopyableMarkdown.on_click)
        self.assertIn("event.ctrl", src)
        self.assertNotIn("is_ctrl", src)
        self.assertNotIn("is_double", src)

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
