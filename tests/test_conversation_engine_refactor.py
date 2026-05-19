import unittest
from types import SimpleNamespace

from conversation_engine import run_model_stream
from model_interface import FakeModelPort


class FakeScroll:
    def __init__(self):
        self.scroll_offset = SimpleNamespace(y=0)
        self.virtual_size = SimpleNamespace(height=10)
        self.region = SimpleNamespace(height=100)
        self.scrolled = False

    def scroll_end(self, animate: bool = False) -> None:
        self.scrolled = True


class FakeChat:
    def __init__(self):
        self.first_message = False
        self.interrupted = False
        self.chunk_updates = []
        self.finished_updates = []
        self.busy = True
        self.crash_calls = 0
        self.scroll = FakeScroll()
        self._on_crash = self.on_crash

    async def on_crash(self) -> None:
        self.crash_calls += 1

    def query_one(self, selector, _widget_type=None):
        assert selector == "#chat"
        return self.scroll

    async def handle_stream_chunk(self, display: str) -> None:
        self.chunk_updates.append(display)

    async def handle_stream_finished(self, display: str) -> None:
        self.finished_updates.append(display)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy


class BrokenPort:
    running = True

    async def send_message(self, text: str):
        raise RuntimeError("boom")
        if False:
            yield text


class TestConversationEngine(unittest.IsolatedAsyncioTestCase):
    async def test_run_model_stream_updates_chunks_and_finishes(self):
        chat = FakeChat()
        port = FakeModelPort(chunks=["Hel", "lo", "!"])

        await run_model_stream(chat, port, "hello")

        self.assertIn("Hel", chat.chunk_updates[0])
        self.assertIn("Hello!", chat.finished_updates[-1])
        self.assertFalse(chat.busy)
        self.assertTrue(chat.scroll.scrolled)

    async def test_run_model_stream_marks_interrupted_output(self):
        chat = FakeChat()
        chat.interrupted = True
        port = FakeModelPort(chunks=["Hello"])

        await run_model_stream(chat, port, "hello")

        self.assertIn("stopped", chat.finished_updates[-1])
        self.assertFalse(chat.interrupted)

    async def test_run_model_stream_invokes_crash_callback_on_failure(self):
        chat = FakeChat()

        await run_model_stream(chat, BrokenPort(), "hello")

        self.assertEqual(chat.crash_calls, 1)


if __name__ == "__main__":
    unittest.main()
