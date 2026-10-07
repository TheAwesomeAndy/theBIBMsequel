# SPDX-License-Identifier: Apache-2.0
"""LaTeX parsing utilities shared by every check.

All helpers operate on a comment-stripped, ``\\input``-expanded document so
that character offsets are stable and can be resolved back to file:line
through the document's source map.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from papercheck.model import Document, Segment, SourceMap

INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^{}]+)\}")
SECTION_PATTERN = re.compile(r"\\(section|subsection|subsubsection|chapter)\*?\s*\{")
# Abbreviations whose trailing period must not end a sentence.
_ABBREVIATIONS = (
    "e.g", "i.e", "et al", "etc", "cf", "vs", "fig", "figs", "eq", "eqs",
    "sec", "tab", "no", "vol", "pp", "approx", "resp", "ca", "dr", "prof",
    "st", "jr", "u.s", "u.k", "ph.d",
)


def strip_comments(text: str) -> str:
    """Blank out LaTeX comments while preserving offsets and line breaks."""
    out: list[str] = []
    for line in text.split("\n"):
        result = []
        index = 0
        while index < len(line):
            char = line[index]
            if char == "%" and (index == 0 or line[index - 1] != "\\"):
                # Keep the line length identical so offsets stay valid.
                result.append(" " * (len(line) - index))
                break
            result.append(char)
            index += 1
        out.append("".join(result))
    return "\n".join(out)


def load_document(main_tex: Path, max_depth: int = 8) -> Document:
    """Read, comment-strip, and recursively expand a LaTeX document."""
    source_map = SourceMap()
    chunks: list[str] = []
    out_pos = 0
    seen: set[Path] = set()

    def emit(path: Path, file_text: str, start: int, end: int) -> None:
        nonlocal out_pos
        if end <= start:
            return
        chunk = file_text[start:end]
        source_map.add(
            Segment(
                out_start=out_pos,
                out_end=out_pos + len(chunk),
                path=str(path),
                file_text=file_text,
                file_offset=start,
            )
        )
        chunks.append(chunk)
        out_pos += len(chunk)

    def expand(path: Path, depth: int) -> None:
        resolved = path.resolve()
        if resolved in seen or depth > max_depth or not path.exists():
            return
        seen.add(resolved)
        file_text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        cursor = 0
        for match in INPUT_PATTERN.finditer(file_text):
            emit(path, file_text, cursor, match.start())
            target = match.group(1).strip()
            child = path.parent / target
            if child.suffix == "":
                child = child.with_suffix(".tex")
            if child.exists():
                chunks.append("\n")
                out_pos_before = out_pos  # noqa: F841 (kept for clarity)
                _pad_newline()
                expand(child, depth + 1)
                chunks.append("\n")
                _pad_newline()
            cursor = match.end()
        emit(path, file_text, cursor, len(file_text))

    def _pad_newline() -> None:
        nonlocal out_pos
        out_pos += 1

    expand(main_tex, 0)
    return Document(main_path=main_tex, text="".join(chunks), source_map=source_map)


def extract_braced(text: str, start: int) -> tuple[str, int] | None:
    """Return (content, end_index_after_closing_brace) for a balanced group.

    ``start`` must point at the opening ``{``.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and text[index - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def command_arg(text: str, command: str) -> tuple[str, int] | None:
    """Find ``\\command{...}`` and return (argument, position of command)."""
    pattern = re.compile(r"\\" + re.escape(command) + r"\s*(\[[^\]]*\]\s*)?\{")
    match = pattern.search(text)
    if not match:
        return None
    extracted = extract_braced(text, match.end() - 1)
    if extracted is None:
        return None
    return extracted[0], match.start()


@dataclass
class Environment:
    name: str
    body: str
    start: int
    end: int


def environments(text: str, *names: str) -> list[Environment]:
    """Find ``\\begin{name}...\\end{name}`` blocks (including starred forms)."""
    found: list[Environment] = []
    for name in names:
        pattern = re.compile(
            r"\\begin\{" + re.escape(name) + r"(\*?)\}(.*?)\\end\{" + re.escape(name) + r"\1\}",
            re.S,
        )
        for match in pattern.finditer(text):
            found.append(
                Environment(name=name + match.group(1), body=match.group(2), start=match.start(), end=match.end())
            )
    found.sort(key=lambda env: env.start)
    return found


@dataclass
class Section:
    level: str
    title: str
    start: int
    body_start: int
    body_end: int
    body: str

    def title_matches(self, *keywords: str) -> bool:
        lowered = self.title.lower()
        return any(keyword.lower() in lowered for keyword in keywords)


def sections(text: str) -> list[Section]:
    """Extract sectioning commands with their body spans.

    A section's body runs until the next sectioning command of the same or
    higher level, or the end of the document body.
    """
    order = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}
    raw: list[tuple[str, str, int, int]] = []
    for match in SECTION_PATTERN.finditer(text):
        extracted = extract_braced(text, match.end() - 1)
        if extracted is None:
            continue
        title, after = extracted
        raw.append((match.group(1), title.strip(), match.start(), after))

    end_document = text.find("\\end{document}")
    doc_end = end_document if end_document >= 0 else len(text)

    result: list[Section] = []
    for index, (level, title, start, body_start) in enumerate(raw):
        body_end = doc_end
        for next_level, _t, next_start, _a in raw[index + 1 :]:
            if order[next_level] <= order[level]:
                body_end = next_start
                break
        else:
            body_end = doc_end
        body_end = max(body_start, min(body_end, doc_end)) if start < doc_end else body_start
        result.append(
            Section(
                level=level,
                title=title,
                start=start,
                body_start=body_start,
                body_end=body_end,
                body=text[body_start:body_end],
            )
        )
    return result


def find_section(all_sections: list[Section], *keywords: str) -> Section | None:
    for section in all_sections:
        if section.title_matches(*keywords):
            return section
    return None


def get_title(text: str) -> tuple[str, int] | None:
    result = command_arg(text, "title")
    if result is None:
        return None
    title, pos = result
    # icml/acl style: \title{Main Title \\ Subtitle}. Strip line breaks.
    title = re.sub(r"\\\\", " ", title)
    return detex(title).strip(), pos


def get_abstract(text: str) -> tuple[str, int] | None:
    for env in environments(text, "abstract", "IEEEabstract"):
        return env.body.strip(), env.start
    result = command_arg(text, "abstract")
    if result is not None:
        return result[0].strip(), result[1]
    return None


def get_keywords(text: str) -> tuple[list[str], int] | None:
    body: str | None = None
    pos = 0
    for env in environments(text, "keywords", "IEEEkeywords"):
        body, pos = env.body, env.start
        break
    if body is None:
        result = command_arg(text, "keywords")
        if result is not None:
            body, pos = result
    if body is None:
        return None
    body = re.sub(r"\\sep\b", ",", body)
    parts = [detex(part).strip() for part in re.split(r"[,;·]", body)]
    return [part for part in parts if part], pos


def detex(text: str) -> str:
    """Reduce LaTeX to plain words for counting and sentence splitting."""
    out = text
    out = re.sub(r"\$\$.*?\$\$", " MATH ", out, flags=re.S)
    out = re.sub(r"\$[^$]*\$", " MATH ", out)
    out = re.sub(r"\\\[.*?\\\]", " MATH ", out, flags=re.S)
    out = re.sub(r"\\begin\{(equation|align|gather|eqnarray)\*?\}.*?\\end\{\1\*?\}", " MATH ", out, flags=re.S)
    out = re.sub(r"\\(cite|citep|citet|ref|autoref|cref|Cref|eqref|label)\s*(\[[^\]]*\])?\{[^{}]*\}", " REF ", out)
    out = re.sub(r"\\(emph|textbf|textit|texttt|underline|mbox|text)\{([^{}]*)\}", r"\2", out)
    out = re.sub(r"\\[a-zA-Z@]+\*?", " ", out)
    out = out.replace("{", " ").replace("}", " ").replace("~", " ")
    return re.sub(r"\s+", " ", out).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']*", text)


def sentences(text: str) -> list[str]:
    """Split plain text into sentences, protecting common abbreviations."""
    protected = text
    for abbreviation in _ABBREVIATIONS:
        pattern = re.compile(r"(?i)\b" + re.escape(abbreviation) + r"\.")
        protected = pattern.sub(lambda m: m.group(0)[:-1] + "․", protected)
    # Protect decimal points and single-letter initials.
    protected = re.sub(r"(\d)\.(\d)", r"\1․\2", protected)
    protected = re.sub(r"\b([A-Z])\.(\s[A-Z])", r"\1․\2", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    restored = [part.replace("․", ".").strip() for part in parts]
    return [part for part in restored if re.search(r"[A-Za-z]", part)]


def cite_keys(text: str) -> list[tuple[str, int]]:
    """All cited keys with the position of the citing command."""
    keys: list[tuple[str, int]] = []
    for match in re.finditer(r"\\[cC]ite[a-zA-Z]*\s*(\[[^\]]*\]\s*)*\{([^{}]*)\}", text):
        for key in match.group(2).split(","):
            key = key.strip()
            if key:
                keys.append((key, match.start()))
    return keys


def find_all(pattern: str | re.Pattern, text: str, flags: int = re.IGNORECASE) -> list[re.Match]:
    compiled = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
    return list(compiled.finditer(text))


def document_body(text: str) -> tuple[str, int]:
    """Return (body, offset) between \\begin{document} and \\end{document}."""
    begin = text.find("\\begin{document}")
    if begin < 0:
        return text, 0
    start = begin + len("\\begin{document}")
    end = text.find("\\end{document}", start)
    if end < 0:
        end = len(text)
    return text[start:end], start
