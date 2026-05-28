import unittest
from typing import AsyncIterator

from tui_main import ChatUI


class StubPort:
    def __init__(self):
        self.running = True

    async def start(self, model_path, options, system_prompt):
        return True

    async def send_message(self, text: str) -> AsyncIterator[str]:
        return
        yield  # pragma: no cover

    async def send_command(self, text: str, timeout: int = 60) -> str | None:
        return None

    async def interrupt(self) -> None:
        pass

    async def stop(self) -> None:
        self.running = False


class TestHelpOverlay(unittest.IsolatedAsyncioTestCase):

    async def test_ctrl_shift_h_not_a_valid_key(self):
        """Verify ctrl+shift+h is not a valid Textual key (terminals can't send it)."""
        from textual.keys import Keys
        all_key_values = {k.value for k in Keys}
        self.assertNotIn(
            "ctrl+shift+h", all_key_values,
            "ctrl+shift+h is not a valid terminal key — "
            "use ctrl+backslash instead"
        )

    async def test_ctrl_backslash_toggles_help(self):
        """Ctrl+\\ should toggle the help overlay."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            help_box = app.query_one("#help-overlay")
            self.assertFalse(help_box.display)

            await pilot.press("ctrl+backslash")
            await pilot.pause()
            self.assertTrue(
                help_box.display,
                "Help overlay should be visible after Ctrl+\\"
            )

            await pilot.press("ctrl+backslash")
            await pilot.pause()
            self.assertFalse(
                help_box.display,
                "Help overlay should be hidden after second Ctrl+\\"
            )

    async def test_escape_closes_help(self):
        """Escape should close the help overlay."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            self.assertTrue(app.query_one("#help-overlay").display)

            await pilot.press("escape")
            await pilot.pause()
            self.assertFalse(app.query_one("#help-overlay").display)

    async def test_help_has_content(self):
        """Help overlay should contain keybinding info."""
        app = ChatUI(port=StubPort())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            content = app.query_one("#help-content")
            text = str(content.render())
            self.assertIn("Copy", text)
            self.assertIn("Select", text)
            self.assertIn("Chat", text)
            self.assertIn("Commands", text)


if __name__ == "__main__":
    unittest.main()
