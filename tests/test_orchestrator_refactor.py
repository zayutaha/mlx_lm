import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from orchestrator import Orchestrator


class FakeInput:
    def __init__(self, text: str):
        self.text = text
        self.focus = Mock()

    def clear(self) -> None:
        self.text = ""


class FakeWidget:
    def __init__(self):
        self.display = False
        self.updated = []
        self.focus = Mock()

    def update(self, value: str) -> None:
        self.updated.append(value)


class FakeChat:
    def __init__(self, text: str = ""):
        self.busy = False
        self.loading = False
        self.interrupted = False
        self.crash_dialog_visible = False
        self.exit = Mock()
        self.handle_stream_text = AsyncMock()
        self.reset_chat = AsyncMock()
        self.clear_chat = AsyncMock()
        self.show_model_selector = Mock()
        self.show_options_selector = Mock()
        self.show_personality_selector = Mock()
        self.show_model_loading = Mock()
        self.show_chat_ui = Mock()
        self.refresh_command_menu = Mock()
        self._set_busy = Mock(side_effect=self._record_busy)
        self._busy_changes = []
        self.widgets = {
            "#input": FakeInput(text),
            "#crash-dialog-container": FakeWidget(),
            "#crash-message": FakeWidget(),
            "#crash-reload": FakeWidget(),
            "#chat-center": SimpleNamespace(display=True),
        }

    def _record_busy(self, value: bool) -> None:
        self.busy = value
        self._busy_changes.append(value)

    def query_one(self, selector):
        return self.widgets[selector]


class FakePort:
    def __init__(self, running: bool = True):
        self.running = running
        self.sent_commands = []
        self.stopped = False

    async def start(self, model_path: str, options: dict, system_prompt: str) -> bool:
        self.running = True
        return True

    async def send_message(self, text: str):
        if False:
            yield text

    async def send_command(self, text: str, timeout: int = 60) -> None:
        self.sent_commands.append(text)

    async def interrupt(self) -> None:
        return None

    async def stop(self) -> None:
        self.running = False
        self.stopped = True


class TestOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def test_handle_submit_clear_resets_chat_and_sends_clear(self):
        chat = FakeChat("/clear")
        port = FakePort(running=True)
        controller = Orchestrator(chat, port=port)

        await controller.handle_submit()

        chat.reset_chat.assert_awaited_once()
        self.assertEqual(port.sent_commands, ["/clear"])
        self.assertEqual(chat._busy_changes[-1], False)

    async def test_handle_submit_starts_stream_task_for_normal_message(self):
        chat = FakeChat("hello")
        port = FakePort(running=True)
        controller = Orchestrator(chat, port=port)
        created = []

        def fake_create_task(coro):
            created.append(coro)
            coro.close()
            return SimpleNamespace(done=lambda: False, cancel=lambda: None)

        with patch("orchestrator.asyncio.create_task", side_effect=fake_create_task):
            await controller.handle_submit()

        chat.handle_stream_text.assert_awaited_once_with("hello")
        self.assertEqual(chat._busy_changes[-1], True)
        self.assertEqual(len(created), 1)

    async def test_handle_crash_from_chat_shows_dialog_in_chat(self):
        chat = FakeChat()
        port = FakePort(running=False)
        controller = Orchestrator(chat, port=port)

        await controller.handle_crash_from_chat()

        self.assertTrue(chat.loading)
        self.assertTrue(chat.crash_dialog_visible)
        self.assertTrue(chat.widgets["#crash-dialog-container"].display)
        self.assertIn("1/3", chat.widgets["#crash-message"].updated[-1])

    async def test_handle_crash_from_chat_reloads_when_not_in_chat(self):
        chat = FakeChat()
        chat.widgets["#chat-center"].display = False
        port = FakePort(running=False)
        controller = Orchestrator(chat, port=port)

        created = []

        def fake_create_task(coro):
            created.append(coro)
            coro.close()
            return SimpleNamespace(done=lambda: False, cancel=lambda: None)

        with patch("orchestrator.asyncio.create_task", side_effect=fake_create_task):
            await controller.handle_crash_from_chat()

        chat.show_model_loading.assert_called_once()
        self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
