from textual.events import Key
from textual.widgets import Static


class PersonalitySelector(Static):
    can_focus = True

    def __init__(self, personalities: list[tuple[str, str]], **kwargs):
        super().__init__(**kwargs)
        self.personalities = personalities
        self.selected_index = 0
        self.render_list()

    def render_list(self):
        lines = ["[bold #f0a500]Select a personality:[/bold #f0a500]\n"]
        for i, (name, description) in enumerate(self.personalities):
            label = name.title()
            if i == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {label}[/bold #f0a500]")
                lines.append(f"  [dim]{description}[/dim]")
            else:
                lines.append(f"  {label}")
                lines.append(f"  [dim]{description}[/dim]")
        lines.append("\n[dim](↑/↓ navigate, Enter select, Esc back, Ctrl+C quit)[/dim]")
        self.update("\n".join(lines))

    async def on_key(self, event: Key) -> None:
        if event.key == "up":
            event.prevent_default()
            self.selected_index = (self.selected_index - 1) % len(self.personalities)
            self.render_list()
        elif event.key == "down":
            event.prevent_default()
            self.selected_index = (self.selected_index + 1) % len(self.personalities)
            self.render_list()
        elif event.key == "enter":
            event.prevent_default()
            selected = self.personalities[self.selected_index][0]
            await self.app.action_personality_selected(selected)
        elif event.key == "escape":
            event.prevent_default()
            await self.app.action_dismiss_personality_selector()
        elif event.key == "ctrl+c":
            event.prevent_default()
            self.app.exit()
