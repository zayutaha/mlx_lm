# 🔴 MUST be first — fixes Textual crash
import textual.keys as tk

_orig = tk.key_to_character
def safe_key_to_character(key):
    if key is None:
        return None
    return _orig(key)
tk.key_to_character = safe_key_to_character


import asyncio
import random
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Middle, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.widgets import Button, Markdown, Static

from model_catalog import list_models
from textual_ui.styles import CHAT_CSS, LOGO, WELCOME_MESSAGES
from textual_ui.latex import format_for_display, strip_prompt_markers
from textual_ui.personas import PERSONALITIES

from tui_chat_input import ChatInput
from tui_loading_spinner import LoadingSpinner
from tui_model_picker import ModelSelector
from tui_model_config_editor import ModelConfigEditor
from tui_options_selector import OptionsSelector
from tui_personality_selector import PersonalitySelector
from tui_slash_command_menu import SlashCommandMenu
from conversation_engine import run_model_stream


class ChatUI(App):
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "reload_model", "Reload Model"),
    ]
    CSS = CHAT_CSS

    def __init__(self, port=None, on_crash=None, on_reload=None, on_quit=None, **kwargs):
        super().__init__(**kwargs)
        self.port = port
        self._on_crash = on_crash
        self._on_reload = on_reload
        self._on_quit = on_quit
        self.loading = False
        self.busy = False
        self.interrupted = False
        self.first_message = True
        self.crash_dialog_visible = False

    def compose(self) -> ComposeResult:
        with Center(id="model-selector-container"):
            yield ModelSelector(list_models({}), id="model-selector")

        with Center(id="personality-selector-container"):
            yield PersonalitySelector(
                [
                    ("default", "Blunt, compact answers with no fake politeness."),
                    ("doctor", "Medical explainer who asks follow-up questions first."),
                    ("historian", "Opinionated historical analysis with sharper language."),
                ],
                id="personality-selector",
            )

        with Center(id="options-selector-container"):
            yield OptionsSelector({}, id="options-selector")

        with Center(id="model-editor-container"):
            yield ModelConfigEditor("", {}, id="model-editor")

        with Center(id="splash-container"):
            yield Static(LOGO, id="splash-logo")
            yield LoadingSpinner(id="load-spinner")

        with Vertical(id="chat-center"):
            yield VerticalScroll(id="chat")

        with Center(id="command-menu-container"):
            yield SlashCommandMenu(id="command-menu")

        with Center(id="input-center"):
            with Horizontal(id="input-card"):
                yield ChatInput(id="input")
                yield Static(" SEND ", id="send-btn")

        with Middle(id="crash-dialog-container"):
            with Vertical(id="crash-dialog", classes="crash-dialog"):
                yield Static("Model crashed. What do you want to do?", id="crash-message", classes="crash-message")
                with Horizontal(classes="crash-buttons"):
                    yield Button("Reload", id="crash-reload", variant="primary")
                    yield Button("Quit", id="crash-quit", variant="error")

    async def on_mount(self):
        self.loading = False
        self.busy = False
        self.interrupted = False
        self.first_message = True
        self.crash_dialog_visible = False

    # ── Public UI control methods (called by orchestrator) ──

    def show_chat_ui(self):
        self.loading = False
        self.query_one("#splash-container").display = False
        self.query_one("#chat-center").display = True
        self.query_one("#input-center").display = True
        self.refresh_command_menu()
        self.call_after_refresh(self._mount_welcome_screen)
        self.query_one("#input").focus()

    def show_loading(self, message="Loading model..."):
        self.loading = True
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        splash = self.query_one("#splash-container")
        splash.display = True
        spinner = self.query_one("#load-spinner", LoadingSpinner)
        spinner.message = message
        spinner.spinner_index = 0
        spinner.update(f"[bold #f0a500]{spinner.SPINNERS[0]} {spinner.message}")

    async def reset_chat(self):
        old_chat = self.query_one("#chat", VerticalScroll)
        await old_chat.remove()
        await self.query_one("#chat-center").mount(VerticalScroll(id="chat"))
        self._mount_welcome_screen()

    # ── Chat UI internals ──

    @property
    def command_menu_visible(self) -> bool:
        return bool(self.query_one("#command-menu-container").display)

    def refresh_command_menu(self) -> None:
        if self.loading or self.busy or not self.query_one("#chat-center").display or not self.query_one("#input-center").display:
            self.query_one("#command-menu-container").display = False
            return
        box = self.query_one("#input", ChatInput)
        container = self.query_one("#command-menu-container")
        menu = self.query_one("#command-menu", SlashCommandMenu)
        container.display = menu.update_matches(box.text)

    def hide_command_menu(self) -> None:
        menu = self.query_one("#command-menu", SlashCommandMenu)
        menu.matches = []
        menu.selected_index = 0
        self.query_one("#command-menu-container").display = False

    def move_command_selection(self, direction: int) -> None:
        menu = self.query_one("#command-menu", SlashCommandMenu)
        if menu.move_selection(direction):
            self.query_one("#command-menu-container").display = True

    def apply_selected_command(self) -> None:
        menu = self.query_one("#command-menu", SlashCommandMenu)
        selected = menu.selected_command()
        if not selected:
            return
        box = self.query_one("#input", ChatInput)
        box.load_text(selected)
        self.hide_command_menu()

    def _mount_welcome_screen(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        existing_ids = {child.id for child in chat.children}
        if "welcome-logo" in existing_ids or "welcome-prompt" in existing_ids:
            return
        welcome = random.choice(WELCOME_MESSAGES)
        chat.mount(Markdown(f"```\n{welcome}\n```", id="welcome-logo", classes="bubble-welcome"))
        chat.mount(Static("How can I help you?", id="welcome-prompt", classes="bubble-prompt"))
        chat.scroll_end(animate=False)

    # ── Action handlers ──

    async def action_submit(self):
        if self.busy or self.loading or not self.port or not self.port.running:
            return

        box = self.query_one("#input", ChatInput)
        user_text = box.text.strip()
        if not user_text:
            return
        box.clear()

        chat = self.query_one("#chat", VerticalScroll)

        if user_text == "/clear":
            await self.reset_chat()
            if self.port and self.port.running:
                try:
                    await self.port.send_command("/clear")
                except Exception:
                    pass
            self._set_busy(False)
            self.refresh_command_menu()
            self.query_one("#input").focus()
            return

        if user_text == "/models":
            await self.show_model_selector()
            return
        if user_text == "/options":
            await self.show_options_selector()
            return
        if user_text == "/personality":
            await self.show_personality_selector()
            return

        await chat.mount(Markdown(user_text, classes="bubble-user"))
        self.current_md = Markdown("▌", classes="bubble-assistant")
        await chat.mount(self.current_md)
        chat.scroll_end(animate=False)

        self._set_busy(True)
        asyncio.create_task(self.run_model(user_text))

    def _set_busy(self, busy: bool):
        self.busy = busy
        btn = self.query_one("#send-btn", Static)
        btn.update(" STOP " if busy else " SEND ")
        btn.set_class(busy, "stopping")

    async def on_static_click(self, event: Click):
        if event.widget.id == "send-btn":
            if self.busy:
                await self.action_interrupt()
            else:
                await self.action_submit()

    async def action_interrupt(self):
        if self.busy and self.port:
            await self.port.interrupt()
            self.interrupted = True

    async def action_quit(self):
        if self.port:
            await self.port.stop()
        self.exit()

    async def on_key(self, event: Key) -> None:
        if event.key == "enter" and self.crash_dialog_visible:
            self.query_one("#crash-reload").press()
            event.prevent_default()
            event.stop()

    async def action_reload_model(self) -> None:
        if self._on_reload:
            await self._on_reload()

    async def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "crash-reload":
            if self._on_reload:
                await self._on_reload()
        elif event.button.id == "crash-quit":
            if self._on_quit:
                await self._on_quit()

    async def run_model(self, user_text: str):
        await run_model_stream(self, user_text)

    async def show_model_selector(self):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#model-selector-container").display = True
        self.query_one("#model-selector").focus()

    async def show_options_selector(self):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#options-selector-container").display = True
        self.query_one("#options-selector").focus()

    async def show_personality_selector(self):
        self.query_one("#chat-center").display = False
        self.query_one("#input-center").display = False
        self.query_one("#command-menu-container").display = False
        self.query_one("#personality-selector-container").display = True
        self.query_one("#personality-selector").focus()
