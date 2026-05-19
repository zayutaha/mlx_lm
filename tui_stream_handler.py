import asyncio

from textual.containers import VerticalScroll
from textual.widgets import Markdown

from tui_commands import TUI_PROMPT_MARKER
from textual_ui.latex import format_for_display, strip_prompt_markers


async def run_model_stream(chat, user_text: str):
    if chat.first_message:
        await asyncio.sleep(2)
        chat.first_message = False

    if not chat.orch.running:
        await chat._handle_crash("")
        return

    user_text = " ".join(user_text.split("\n"))

    if not await chat.orch.send(user_text):
        await chat._handle_crash("")
        return

    buf = ""
    last_update = 0
    chat_widget = chat.query_one("#chat", VerticalScroll)
    thinking_enabled = user_text.startswith("/think")

    def get_display_text(buffer):
        last_end = buffer.rfind("</think>")
        if last_end >= 0:
            return buffer[last_end + len("</think>"):].strip()
        return ""

    spinner_index = 0
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    while True:
        try:
            chunk = await asyncio.wait_for(
                chat.orch.runner.proc.stdout.read(256), timeout=0.05
            )
        except asyncio.TimeoutError:
            if chat.interrupted:
                break
            continue
        except Exception:
            await chat._handle_crash("")
            return

        if not chunk:
            await chat._handle_crash("")
            return

        buf += chunk.decode(errors="ignore")

        if buf.endswith(TUI_PROMPT_MARKER):
            buf = buf[: -len(TUI_PROMPT_MARKER)]
            break

        now = asyncio.get_event_loop().time()
        if now - last_update > 0.05:
            if thinking_enabled:
                if "</think>" not in buf:
                    spinner_index = (spinner_index + 1) % len(spinner_frames)
                    await chat.current_md.update(f"Thinking... {spinner_frames[spinner_index]}")
                else:
                    display = strip_prompt_markers(get_display_text(buf))
                    if display:
                        await chat.current_md.update(f"{format_for_display(display)} ▌")
            else:
                display = strip_prompt_markers(buf)
                if display:
                    await chat.current_md.update(f"{format_for_display(display)} ▌")
            last_update = now

    if chat.interrupted:
        remaining = await chat.orch.read_until_prompt(timeout=10)
        if remaining:
            buf += remaining

    if thinking_enabled:
        display = strip_prompt_markers(get_display_text(buf))
    else:
        display = strip_prompt_markers(buf)

    if chat.interrupted:
        display += "\n\n*— stopped*"
        chat.interrupted = False

    try:
        await chat.current_md.update(format_for_display(display))
    except Exception as e:
        await chat.current_md.update(f'<error: {e}>')
        return

    scroll_y = chat_widget.scroll_offset.y
    virtual_h = chat_widget.virtual_size.height
    widget_h = chat_widget.region.height
    if virtual_h <= widget_h:
        chat_widget.scroll_end(animate=False)
    else:
        max_scroll_y = virtual_h - widget_h
        if max_scroll_y - scroll_y <= 50:
            chat_widget.scroll_end(animate=False)
    chat._set_busy(False)
