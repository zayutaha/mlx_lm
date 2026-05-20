from textual.widgets import Static

from settings_store import SLASH_COMMANDS


class SlashCommandMenu(Static):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matches: list[tuple[str, str]] = []
        self.selected_index = 0

    def update_matches(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized.startswith("/"):
            self.matches = []
            self.selected_index = 0
            self.display = False
            return False

        if normalized == "/":
            self.matches = list(SLASH_COMMANDS.items())
        else:
            self.matches = [
                (command, description)
                for command, description in SLASH_COMMANDS.items()
                if command.startswith(normalized)
            ]
        if not self.matches:
            self.selected_index = 0
            self.display = False
            return False

        self.selected_index = min(self.selected_index, len(self.matches) - 1)
        self.render_list()
        self.display = True
        return True

    def render_list(self) -> None:
        lines = ["[bold #f0a500]Commands[/bold #f0a500]\n"]
        for index, (command, description) in enumerate(self.matches):
            if index == self.selected_index:
                lines.append(f"[bold #f0a500]❯ {command}[/bold #f0a500]")
            else:
                lines.append(f"  [bold]{command}[/bold]")
            lines.append(f"  [dim]{description}[/dim]")
        lines.append("\n[dim](↑/↓ navigate, Enter select)[/dim]")
        self.update("\n".join(lines))

    def move_selection(self, direction: int) -> bool:
        if not self.matches:
            return False
        self.selected_index = (self.selected_index + direction) % len(self.matches)
        self.render_list()
        return True

    def selected_command(self) -> str | None:
        if not self.matches:
            return None
        return self.matches[self.selected_index][0]
