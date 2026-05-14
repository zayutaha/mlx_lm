"""Simple markdown → Textual/Rich markup renderer.

Handles: **bold**, *italic*, `code`, # headings, --- rules,
> quotes, -/* lists, 1. lists, ``` fences, paragraphs, line breaks.
Outputs Rich markup strings compatible with Textual widgets.
"""

import re


def render(text: str) -> str:
    """Convert markdown text to Rich markup string."""
    text = _render_blocks(text)
    return text


# ── Block-level rendering ──────────────────────────────────────────────


def _render_blocks(text: str) -> str:
    """Split into blocks, render each, rejoin with double newlines."""
    blocks = _split_blocks(text)
    rendered = []
    in_code_fence = False
    code_lang = ""

    for block in blocks:
        # Check for code fence open/close
        fence_m = re.match(r'^```(\w*)$', block.strip())
        if fence_m:
            if not in_code_fence:
                in_code_fence = True
                code_lang = fence_m.group(1)
                rendered.append(block)  # pass through as-is
                continue
            else:
                in_code_fence = False
                rendered.append(block)  # closing fence
                continue

        if in_code_fence:
            rendered.append(block)  # code content — pass through
            continue

        block = block.strip()
        if not block:
            continue

        # Horizontal rule
        if re.match(r'^[-*_]{3,}$', block):
            rendered.append("[dim]────────────────────[/]")
            continue

        # Blockquote
        if block.startswith("> "):
            lines = block.split("\n")
            quoted = []
            for line in lines:
                content = line.removeprefix("> ").removeprefix(">")
                content = _render_inline(content.strip())
                quoted.append(f"[dim]│[/] {content}")
            rendered.append("\n".join(quoted))
            continue

        # Heading
        heading_m = re.match(r'^(#{1,6})\s+', block.split('\n')[0])
        if heading_m:
            lines = block.split("\n")
            headings = []
            for line in lines:
                hm = re.match(r'^(#{1,6})\s+(.+)$', line)
                if hm:
                    lvl = len(hm.group(1))
                    content = _render_inline(hm.group(2).strip())
                    if lvl <= 4:
                        headings.append(f"[bold]{content}[/]")
                    else:
                        headings.append(f"[italic]{content}[/]")
                else:
                    # Not a heading — fall through to regular paragraph
                    headings = None
                    break
            if headings is not None:
                rendered.append("\n\n".join(headings))
                continue

        # Unordered list
        if re.match(r'^[\s]*[-*+]\s+', block):
            lines = block.split("\n")
            items = []
            for line in lines:
                item_m = re.match(r'^[\s]*[-*+]\s+(.*)', line)
                if item_m:
                    content = _render_inline(item_m.group(1))
                    items.append(f"  • {content}")
            rendered.append("\n".join(items))
            continue

        # Ordered list
        if re.match(r'^[\s]*\d+\.\s+', block):
            lines = block.split("\n")
            items = []
            for i, line in enumerate(lines):
                item_m = re.match(r'^[\s]*\d+\.\s+(.*)', line)
                if item_m:
                    content = _render_inline(item_m.group(1))
                    items.append(f"  {i+1}. {content}")
            rendered.append("\n".join(items))
            continue

        # Regular paragraph — convert single newlines to line breaks
        block = re.sub(r'(?<!\n)\n(?!\n)', '  \n', block)
        rendered.append(_render_inline(block))

    return "\n\n".join(rendered)


def _split_blocks(text: str) -> list[str]:
    """Split into blocks at double newlines, preserving code fences."""
    # Use \n\n as paragraph separator, but don't split inside code fences
    lines = text.split("\n")
    blocks = []
    current = []
    in_fence = False

    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue

        if not in_fence and line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
            continue

        current.append(line)

    if current:
        blocks.append("\n".join(current))

    return blocks


# ── Inline rendering ───────────────────────────────────────────────────


def _render_inline(text: str) -> str:
    """Render inline markdown → Rich markup (bold, italic, code, links)."""
    # Escape existing brackets first
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")

    # Images ![alt](url) — strip, return alt text italic
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', text)

    # Links [text](url) — preserve text
    text = re.sub(r'(?<!\\)\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # Inline code `code`
    text = re.sub(r'`([^`]+)`', r'[bold notrailing]`\1`[/]', text)

    # Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'[strike]\1[/]', text)

    # Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'[bold]\1[/]', text)
    text = re.sub(r'__(.+?)__', r'[bold]\1[/]', text)

    # Italic *text* or _text_ (single — careful with underscores in math)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'[italic]\1[/]', text)
    # Only convert _ for italic when surrounded by word boundaries
    text = re.sub(r'(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)', r'[italic]\1[/]', text)

    return text
