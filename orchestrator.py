import asyncio
from pathlib import Path

from model_catalog import list_models
from model_interface import MLXSubprocessAdapter, ModelPort
from settings_store import (
    DEFAULT_MODEL_OPTIONS,
    load_model_configs,
    load_saved_model_options,
    normalize_model_options,
    save_model_configs,
    save_model_options,
    MODEL_CONFIGS_PATH,
)
from model_lifecycle import build_model_command


class Orchestrator:
    def __init__(self, chat_ui):
        self.chat = chat_ui
        self.port: ModelPort = MLXSubprocessAdapter()
        self.selected_model = None
        self.selected_personality = "default"
        self.model_options = load_saved_model_options()
        self.crash_count = 0
        self.max_crashes = 3
        self.reloading = False

    @property
    def current_system_prompt(self) -> str:
        from textual_ui.personas import PERSONALITIES
        return PERSONALITIES.get(self.selected_personality, PERSONALITIES["default"])

    async def handle_model_selected(self, model_name: str):
        self.selected_model = model_name
        configs = load_model_configs()
        model_cfg = configs.get(model_name, {})
        self.selected_personality = model_cfg.get("personality", "default")
        self.chat.show_loading(f"Loading {model_name}...")
        await self.chat.reset_chat()
        await self.port.stop()
        await self._load_model()

    async def _load_model(self):
        model_path = str(Path.home() / ".omlx" / "models" / self.selected_model)
        ok = await self.port.start(model_path, self.model_options, self.current_system_prompt)
        if ok:
            self.crash_count = 0
            self.chat.show_chat_ui()
        else:
            await self._handle_crash("Model failed to initialize")

    async def handle_reload(self):
        if self.port.running:
            return
        if self.chat.loading or self.reloading:
            return

        self.chat._set_busy(False)  # should be a public method
        self.chat.crash_dialog_visible = False
        self.chat.query_one("#crash-dialog-container").display = False
        await self.chat.reset_chat()
        self.crash_count = 0
        self.reloading = True
        self.chat.show_loading("Reloading model...")
        asyncio.create_task(self._load_model())

    async def handle_crash_from_chat(self):
        self.chat._set_busy(False)
        self.chat.loading = True
        self.crash_count += 1

        if self.crash_count >= self.max_crashes:
            self.chat.exit("Too many crashes, giving up")
            return

        if self.chat.query_one("#chat-center").display == False:
            self.reloading = True
            self.chat.show_loading(f"Reloading model (crash #{self.crash_count})...")
            await self.port.stop()
            asyncio.create_task(self._load_model())
            return

        self.reloading = True
        self.chat.crash_dialog_visible = True
        self.chat.query_one("#crash-dialog-container").display = True
        self.chat.query_one("#crash-message").update(
            f"Model crashed (attempt {self.crash_count}/{self.max_crashes}). Reload or quit?"
        )
        self.chat.query_one("#crash-reload").focus()

    async def handle_crash_reload(self):
        self.chat.crash_dialog_visible = False
        self.chat.query_one("#crash-dialog-container").display = False
        self.chat.show_loading(f"Reloading model (crash #{self.crash_count})...")
        await self.port.stop()
        asyncio.create_task(self._load_model())

    async def handle_quit(self):
        self.chat.exit("Model crashed")

    async def handle_options_changed(self, options: dict):
        self.model_options = normalize_model_options(options)
        save_model_options(self.model_options)

    async def handle_model_edit_save(self, model_name: str, config: dict):
        configs = load_model_configs()
        configs[model_name] = config
        save_model_configs(configs)
