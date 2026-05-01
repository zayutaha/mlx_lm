import asyncio
import re
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Markdown, TextArea
from textual.containers import VerticalScroll
from textual.events import Key
from pylatexenc.latex2text import LatexNodes2Text
from sympy import sympify, pretty

BASE_CMD = [
    "uv", "run", "mlx_lm.chat",
    "--model", "/Users/zayaantaha/.omlx/models/Qwwwen",
    "--mtp",
    "--turbo-kv-bits", "3",
    "--turbo-fp16-layers", "2",
    "--max-tokens", "16000",
    "--chat-template-args", '{"enable_thinking":false}',
]

latex_converter = LatexNodes2Text()

def normalize_latex(text: str) -> str:
    # normalize double backslash line breaks only in math contexts, not globally
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    
    fixes = ["frac", "int", "sum", "sqrt", "sin", "cos", "tan", "log", "ln"]
    for cmd in fixes:
        # only match bare word NOT preceded by \ and NOT inside a larger word
        text = re.sub(rf'(?<!\\)\b{cmd}\b', rf'\\{cmd}', text)
    return text

def try_sympy(expr: str) -> str:
    try:
        return f"\n```\n{pretty(sympify(expr))}\n```\n"
    except Exception:
        try:
            return latex_converter.latex_to_text(expr)
        except Exception:
            return expr

def transform_math(text: str) -> str:
    text = normalize_latex(text)
    text = re.sub(r"\$\$(.*?)\$\$", lambda m: try_sympy(m.group(1).strip()), text, flags=re.DOTALL)
    text = re.sub(r"\$(.*?)\$", lambda m: try_sympy(m.group(1).strip()), text)
    return text

def strip_prompt_markers(text: str) -> str:
    """Remove >> prompt markers and info lines from display."""
    lines = text.splitlines()
    clean = []
    for line in lines:
        if line.strip().startswith(">>") or line.startswith("[INFO]"):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


class ChatInput(TextArea):
    async def _on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            await self.app.action_submit()
        else:
            await super()._on_key(event)


class ChatUI(App):
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #chat { height: 1fr; padding: 1 2; }
    .bubble { margin-bottom: 1; padding: 0 1; }
    #input { dock: bottom; height: 6; border-top: solid $primary; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat")
        yield ChatInput(id="input")
        yield Footer()

    async def on_mount(self):
        self.proc = await asyncio.create_subprocess_exec(
            *BASE_CMD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.busy = False
        # drain startup banner until first ">> "
        await self._read_until_prompt()

    async def _read_until_prompt(self) -> str:
        """Read stdout until mlx_lm.chat shows '>> '."""
        buf = ""
        while True:
            ch = await self.proc.stdout.read(1)
            if not ch:
                break
            buf += ch.decode(errors="ignore")
            if buf.endswith(">> "):
                break
        return buf

    async def action_submit(self):
        if self.busy:
            return
        box = self.query_one("#input", ChatInput)
        user_text = box.text.strip().replace("\n", " ")
        box.clear()
        if not user_text:
            return

        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Markdown(f"**You:** {user_text}", classes="bubble"))
        self.current_md = Markdown("**Assistant:** ▌", classes="bubble")
        await chat.mount(self.current_md)
        chat.scroll_end()

        asyncio.create_task(self.run_model(user_text))

    async def _read_until_prompt(self) -> str:
        buf = ""
        while True:
            chunk = await self.proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk.decode(errors="ignore")
            if buf.endswith(">> "):
                break
        return buf

    async def run_model(self, user_text: str):
        self.busy = True
        self.proc.stdin.write((user_text + "\n").encode())
        await self.proc.stdin.drain()

        buf = ""
        last_update = 0.0

        while True:
            chunk = await self.proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk.decode(errors="ignore")
            if buf.endswith(">> "):
                break

            now = asyncio.get_event_loop().time()
            # update at most ~20fps — smooth, no flicker
            if now - last_update > 0.05:
                display = strip_prompt_markers(transform_math(buf))
                await self.current_md.update(f"**Assistant:**\n\n{display}")
                self.query_one("#chat", VerticalScroll).scroll_end()
                last_update = now

        # final render — clean, no cursor
        display = strip_prompt_markers(transform_math(buf))
        await self.current_md.update(f"**Assistant:**\n\n{display}")
        self.query_one("#chat", VerticalScroll).scroll_end()
        self.busy = False

if __name__ == "__main__":
    ChatUI().run()
