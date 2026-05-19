from textual.events import Key
from textual.widgets import Static

from settings_store import OPTION_SPECS, normalize_model_options


class ModelConfigEditor(Static):
    can_focus = True

    def __init__(self, model_name: str, config: dict, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.options = dict(config.get("options", {}))
        self.personality = config.get("personality", "default")
        self.selected_index = 0
        self._items = self._build_items()
        self.render_list()

    def _build_items(self) -> list[dict]:
        items = []
        for spec in OPTION_SPECS:
            items.append({
                "type": "option",
                "key": spec["key"],
                "label": spec["label"],
                "choices": spec["choices"],
                "description": spec["description"],
                "value": self.options.get(spec["key"]),
            })
        items.append({
            "type": "personality",
            "key": "personality",
            "label": "Default personality",
            "choices": ["default", "doctor", "historian"],
            "description": "Personality applied when this model is loaded.",
            "value": self.personality,
        })
        return items

    def _format_value(self, value: object) -> str:
        if value is None:
            return "Auto"
        if value is True:
            return "On"
        if value is False:
            return "Off"
        return str(value)

    def render_list(self):
        lines = [f"[bold #f0a500]Config: {self.model_name}[/bold #f0a500]\n"]
        for idx, item in enumerate(self._items):
            label = item["label"]
            value = self._format_value(item["value"])
            desc = item["description"]
            if idx == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {label}: {value}[/bold #f0a500]")
            else:
                lines.append(f"  {label}: {value}")
            lines.append(f"  [dim]{desc}[/dim]")
        lines.append("\n[dim](↑/↓ navigate, ←/→ change, Esc close)[/dim]")
        self.update("\n".join(lines))

    def change_value(self, direction: int):
        item = self._items[self.selected_index]
        choices = item["choices"]
        current = item["value"]
        try:
            idx = choices.index(current)
        except ValueError:
            idx = 0
        idx = (idx + direction) % len(choices)
        item["value"] = choices[idx]
        self.render_list()

    async def on_key(self, event: Key):
        if event.key == "up":
            event.prevent_default()
            self.selected_index = (self.selected_index - 1) % len(self._items)
            self.render_list()
        elif event.key == "down":
            event.prevent_default()
            self.selected_index = (self.selected_index + 1) % len(self._items)
            self.render_list()
        elif event.key == "left":
            event.prevent_default()
            self.change_value(-1)
        elif event.key == "right":
            event.prevent_default()
            self.change_value(1)
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_model_editor_save(self._collect())
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()

    def _collect(self) -> dict:
        options = {}
        personality = "default"
        for item in self._items:
            if item["type"] == "option":
                options[item["key"]] = item["value"]
            elif item["type"] == "personality":
                personality = item["value"]
        return {
            "options": normalize_model_options(options),
            "personality": personality,
        }
