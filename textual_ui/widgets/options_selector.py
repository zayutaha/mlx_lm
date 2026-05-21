from textual.events import Key
from textual.widgets import Static

from settings_store import OPTION_SPECS


class OptionsSelector(Static):
    can_focus = True

    def __init__(self, options: dict[str, object], **kwargs):
        super().__init__(**kwargs)
        self.options = dict(options)
        self.selected_index = 0
        self._editing = False
        self._edit_buffer = ""
        self.render_list()

    def set_options(self, options: dict[str, object]) -> None:
        self.options = dict(options)
        self.selected_index = min(self.selected_index, len(OPTION_SPECS) - 1)
        self.render_list()

    def _format_value(self, value: object) -> str:
        if value is None:
            return "Auto"
        if value is True:
            return "On"
        if value is False:
            return "Off"
        return str(value)

    def _parse_value(self, raw: str, spec: dict) -> object | None:
        raw = raw.strip()
        if not raw or raw.lower() in ("auto", "none"):
            return None
        choices = spec["choices"]
        non_none = [c for c in choices if c is not None]
        sample = non_none[0] if non_none else str
        try:
            if isinstance(sample, bool):
                return raw.lower() in ("true", "on", "1", "yes")
            if isinstance(sample, int):
                v = int(raw)
                return v
            if isinstance(sample, float):
                return float(raw)
            return raw
        except (ValueError, TypeError):
            return None

    def render_list(self) -> None:
        lines = ["[bold #f0a500]Launch options:[/bold #f0a500]\n"]
        for index, spec in enumerate(OPTION_SPECS):
            key = spec["key"]
            label = spec["label"]
            description = spec["description"]
            if self._editing and index == self.selected_index:
                val_display = f"[reverse]{self._edit_buffer} [/reverse]"
            else:
                val_display = self._format_value(self.options.get(key))
            if index == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {label}: {val_display}[/bold #f0a500]")
            else:
                lines.append(f"  {label}: {val_display}")
            lines.append(f"  [dim]{description}[/dim]")
        if self._editing:
            lines.append("\n[dim](Enter confirm, Esc cancel, Backspace delete)[/dim]")
        else:
            lines.append("\n[dim](↑/↓ navigate, ←/→ change, [bold]e[/bold] edit, Enter apply, Esc back)[/dim]")
        self.update("\n".join(lines))

    def change_value(self, direction: int) -> None:
        spec = OPTION_SPECS[self.selected_index]
        choices = spec["choices"]
        current = self.options.get(spec["key"])
        try:
            index = choices.index(current)
        except ValueError:
            index = 0
        index = (index + direction) % len(choices)
        self.options[spec["key"]] = choices[index]
        self.render_list()

    async def _commit_edit(self) -> None:
        spec = OPTION_SPECS[self.selected_index]
        parsed = self._parse_value(self._edit_buffer, spec)
        if parsed is not None:
            self.options[spec["key"]] = parsed
        self._editing = False
        self.render_list()

    def _cancel_edit(self) -> None:
        self._editing = False
        self.render_list()

    async def on_key(self, event: Key) -> None:
        if self._editing:
            if event.key == "enter":
                event.prevent_default()
                await self._commit_edit()
            elif event.key == "escape":
                event.prevent_default()
                self._cancel_edit()
            elif event.key == "backspace":
                event.prevent_default()
                self._edit_buffer = self._edit_buffer[:-1]
                self.render_list()
            elif event.key == "full_stop":
                event.prevent_default()
                self._edit_buffer += "."
                self.render_list()
            elif event.key == "hyphen" or event.key == "minus":
                event.prevent_default()
                self._edit_buffer += "-"
                self.render_list()
            elif event.key not in ("up", "down", "left", "right", "enter", "escape", "backspace", "tab", "ctrl+c", "ctrl+d", "ctrl+z", "full_stop", "hyphen", "minus"):
                event.prevent_default()
                self._edit_buffer += event.key
                self.render_list()
            return

        if event.key == "up":
            event.prevent_default()
            self.selected_index = (self.selected_index - 1) % len(OPTION_SPECS)
            self.render_list()
        elif event.key == "down":
            event.prevent_default()
            self.selected_index = (self.selected_index + 1) % len(OPTION_SPECS)
            self.render_list()
        elif event.key == "left":
            event.prevent_default()
            self.change_value(-1)
        elif event.key == "right":
            event.prevent_default()
            self.change_value(1)
        elif event.key == "enter":
            event.prevent_default()
            await self.app.action_options_selected(dict(self.options))
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_options_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()
        elif event.key == "e":
            event.prevent_default()
            self._editing = True
            self._edit_buffer = ""
            self.render_list()
