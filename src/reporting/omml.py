"""LaTeX to Office Math (OMML), for the subset the equations reference uses (T100.4).

The thesis wants equations it can edit in Word's equation editor, which means
native ``<m:oMath>`` elements rather than pictures of formulas. The usual route
-- LaTeX to MathML to OMML through Microsoft's ``MML2OMML.XSL`` -- needs a file
that ships with Office, is not redistributable, and is absent on the CI runner.
So the fifteen formulas of blueprint section 11 are converted here, from the
same LaTeX the dashboard renders with KaTeX: one source, two renderings, and no
second hand-typed copy of any formula.

Supported: groups ``{}``, ``^`` and ``_`` (alone or together), ``\\frac`` /
``\\dfrac`` / ``\\tfrac``, ``\\sqrt``, ``\\sum`` with limits, ``\\left``/``\\right``
delimiters, ``\\text`` / ``\\mathrm``, ``\\hat``, the function names and Greek
letters the equations use, and plain characters. Anything else raises
:class:`OmmlError` -- an unsupported command must fail the export rather than
leak a backslash into the document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

__all__ = ["M_NAMESPACE", "OmmlError", "latex_to_omml"]

M_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"

_GREEK = {
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "delta": "\u03b4",
    "epsilon": "\u03b5",
    "mu": "\u03bc",
    "pi": "\u03c0",
    "sigma": "\u03c3",
    "omega": "\u03c9",
    "Sigma": "\u03a3",
    "Omega": "\u03a9",
}
_SYMBOLS = {"in": "\u2208", "cdot": "\u00b7", "times": "\u00d7", "infty": "\u221e"}
_FUNCTIONS = {"log", "ln", "exp", "max", "min", "arg", "sin", "cos"}
_FRACTIONS = {"frac", "dfrac", "tfrac"}
_TEXT = {"text", "mathrm", "operatorname"}
_SPACES = {",", ";", " ", "!", "quad", "qquad"}
_NARY = {"sum": "\u2211", "prod": "\u220f"}


class OmmlError(ValueError):
    """The LaTeX uses something this converter does not implement."""


@dataclass
class _Node:
    kind: str
    text: str = ""
    upright: bool = False
    parts: list[list[_Node]] = field(default_factory=list)


class _Parser:
    def __init__(self, source: str) -> None:
        self.s = source
        self.i = 0

    # -- low level -----------------------------------------------------------
    def _peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def _skip_space(self) -> None:
        while self._peek().isspace():
            self.i += 1

    def _command(self) -> str:
        """Read ``\\name`` (letters) or ``\\c`` (one non-letter), cursor on the backslash."""
        self.i += 1
        start = self.i
        while self._peek().isalpha():
            self.i += 1
        if self.i == start:
            self.i += 1
        return self.s[start : self.i]

    def _raw_group(self) -> str:
        self._skip_space()
        if self._peek() != "{":
            raise OmmlError("expected '{' at " + str(self.i) + " in " + repr(self.s))
        depth, start = 0, self.i + 1
        while self.i < len(self.s):
            char = self.s[self.i]
            depth += char == "{"
            depth -= char == "}"
            self.i += 1
            if depth == 0:
                return self.s[start : self.i - 1]
        raise OmmlError("unbalanced braces in " + repr(self.s))

    # -- grammar -------------------------------------------------------------
    def sequence(self, stop: str = "") -> list[_Node]:
        nodes: list[_Node] = []
        while True:
            self._skip_space()
            char = self._peek()
            if not char:
                if stop:
                    raise OmmlError("missing " + repr(stop) + " in " + repr(self.s))
                return nodes
            if stop == "}" and char == "}":
                self.i += 1
                return nodes
            if stop == "right" and self.s.startswith("\\right", self.i):
                return nodes
            if char in "^_":
                if not nodes:
                    raise OmmlError("a script with no base in " + repr(self.s))
                nodes[-1] = self._scripts(nodes[-1])
                continue
            node = self._atom()
            if node is None:
                continue
            if node.kind == "nary":
                node = self._nary(node)
                node.parts.append(self.sequence(stop))  # the operand runs to the end
                nodes.append(node)
                return nodes
            nodes.append(node)

    def _script_argument(self) -> list[_Node]:
        self._skip_space()
        if self._peek() == "{":
            self.i += 1
            return self.sequence("}")
        node = self._atom()
        if node is None:
            raise OmmlError("empty script in " + repr(self.s))
        return [node]

    def _scripts(self, base: _Node) -> _Node:
        sub: list[_Node] | None = None
        sup: list[_Node] | None = None
        while True:
            self._skip_space()
            char = self._peek()
            if char == "_" and sub is None:
                self.i += 1
                sub = self._script_argument()
            elif char == "^" and sup is None:
                self.i += 1
                sup = self._script_argument()
            else:
                break
        if sub is not None and sup is not None:
            return _Node("subsup", parts=[[base], sub, sup])
        if sub is not None:
            return _Node("sub", parts=[[base], sub])
        return _Node("sup", parts=[[base], sup or []])

    def _nary(self, node: _Node) -> _Node:
        sub: list[_Node] = []
        sup: list[_Node] = []
        while True:
            self._skip_space()
            if self._peek() == "_":
                self.i += 1
                sub = self._script_argument()
            elif self._peek() == "^":
                self.i += 1
                sup = self._script_argument()
            else:
                break
        node.parts = [sub, sup]
        return node

    def _delimiter(self) -> str:
        self._skip_space()
        if self._peek() == "\\":
            name = self._command()
            if name in ("{", "}", "|"):
                return name
            if name == "lvert" or name == "rvert":
                return "|"
            raise OmmlError("unsupported delimiter \\" + name)
        char = self._peek()
        self.i += 1
        return "" if char == "." else char

    def _atom(self) -> _Node | None:
        char = self._peek()
        if char == "{":
            self.i += 1
            return _Node("group", parts=[self.sequence("}")])
        if char == "}":
            raise OmmlError("unexpected '}' in " + repr(self.s))
        if char != "\\":
            self.i += 1
            return _Node("run", char)

        name = self._command()
        if name in _SPACES:
            return None
        if name in _FRACTIONS:
            return _Node("frac", parts=[self._group(), self._group()])
        if name == "sqrt":
            return _Node("rad", parts=[self._group()])
        if name in _NARY:
            return _Node("nary", _NARY[name])
        if name == "left":
            begin = self._delimiter()
            body = self.sequence("right")
            self._command()  # \right
            return _Node("delim", parts=[body], text=begin + "\x00" + self._delimiter())
        if name in _TEXT:
            return _Node("run", self._raw_group(), upright=True)
        if name == "hat":
            return _Node("acc", "\u0302", parts=[self._group()])
        if name in _FUNCTIONS:
            return _Node("run", name, upright=True)
        if name in _GREEK:
            return _Node("run", _GREEK[name])
        if name in _SYMBOLS:
            return _Node("run", _SYMBOLS[name], upright=True)
        if name in ("{", "}", "|", "%", "&", "#", "_"):
            return _Node("run", name)
        raise OmmlError("unsupported LaTeX command \\" + name + " in " + repr(self.s))

    def _group(self) -> list[_Node]:
        self._skip_space()
        if self._peek() == "{":
            self.i += 1
            return self.sequence("}")
        node = self._atom()
        return [] if node is None else [node]


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------


def _attr(value: str) -> str:
    return escape(value, {'"': "&quot;"})


def _merge(nodes: list[_Node]) -> list[_Node]:
    """Join adjacent plain characters of one style into one run."""
    merged: list[_Node] = []
    for node in nodes:
        last = merged[-1] if merged else None
        if (
            node.kind == "run"
            and last is not None
            and last.kind == "run"
            and last.upright == node.upright
            and len(node.text) == 1
            and not node.upright
        ):
            last.text += node.text
        else:
            merged.append(_Node(node.kind, node.text, node.upright, node.parts))
    return merged


def _emit(nodes: list[_Node]) -> str:
    return "".join(_emit_one(node) for node in _merge(nodes))


def _emit_one(node: _Node) -> str:
    if node.kind == "run":
        style = '<m:rPr><m:sty m:val="p"/></m:rPr>' if node.upright else ""
        return "<m:r>" + style + '<m:t xml:space="preserve">' + escape(node.text) + "</m:t></m:r>"
    if node.kind == "group":
        return _emit(node.parts[0])
    if node.kind == "frac":
        return (
            "<m:f><m:num>"
            + _emit(node.parts[0])
            + "</m:num><m:den>"
            + _emit(node.parts[1])
            + "</m:den></m:f>"
        )
    if node.kind == "rad":
        return (
            '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>'
            + _emit(node.parts[0])
            + "</m:e></m:rad>"
        )
    if node.kind == "nary":
        sub, sup, body = node.parts
        hide = ('<m:subHide m:val="1"/>' if not sub else "") + (
            '<m:supHide m:val="1"/>' if not sup else ""
        )
        return (
            '<m:nary><m:naryPr><m:chr m:val="'
            + _attr(node.text)
            + '"/>'
            + hide
            + "</m:naryPr><m:sub>"
            + _emit(sub)
            + "</m:sub><m:sup>"
            + _emit(sup)
            + "</m:sup><m:e>"
            + _emit(body)
            + "</m:e></m:nary>"
        )
    if node.kind == "delim":
        begin, end = node.text.split("\x00")
        return (
            '<m:d><m:dPr><m:begChr m:val="'
            + _attr(begin)
            + '"/><m:endChr m:val="'
            + _attr(end)
            + '"/></m:dPr><m:e>'
            + _emit(node.parts[0])
            + "</m:e></m:d>"
        )
    if node.kind == "acc":
        return (
            '<m:acc><m:accPr><m:chr m:val="'
            + _attr(node.text)
            + '"/></m:accPr><m:e>'
            + _emit(node.parts[0])
            + "</m:e></m:acc>"
        )
    if node.kind in ("sub", "sup"):
        tag = "sSub" if node.kind == "sub" else "sSup"
        return (
            "<m:"
            + tag
            + "><m:e>"
            + _emit(node.parts[0])
            + "</m:e><m:"
            + node.kind
            + ">"
            + _emit(node.parts[1])
            + "</m:"
            + node.kind
            + "></m:"
            + tag
            + ">"
        )
    if node.kind == "subsup":
        return (
            "<m:sSubSup><m:e>"
            + _emit(node.parts[0])
            + "</m:e><m:sub>"
            + _emit(node.parts[1])
            + "</m:sub><m:sup>"
            + _emit(node.parts[2])
            + "</m:sup></m:sSubSup>"
        )
    raise OmmlError("cannot emit node " + node.kind)  # pragma: no cover - parser bug


def latex_to_omml(latex: str, *, display: bool = True) -> str:
    """One formula as an OMML fragment with its namespace declared.

    ``display`` wraps it in ``<m:oMathPara>`` (its own centred paragraph);
    otherwise a bare ``<m:oMath>`` for use inside running text or a table cell.
    """
    body = "<m:oMath>" + _emit(_Parser(latex).sequence()) + "</m:oMath>"
    if display:
        return '<m:oMathPara xmlns:m="' + M_NAMESPACE + '">' + body + "</m:oMathPara>"
    return body.replace("<m:oMath>", '<m:oMath xmlns:m="' + M_NAMESPACE + '">', 1)
