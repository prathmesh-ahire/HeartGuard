"""T112.6 -- the bundle budget, measured from the exported pages themselves.

`next build` prints a "First Load JS" column and everyone reads it once. This
measures the same thing from `frontend/out/`, the artifact that actually ships:
for every exported page it gzips exactly the scripts that page's HTML asks the
browser to fetch, and **fails** when a budget is exceeded. The number becomes a
gate instead of a line in a log.

`out/` rather than `.next/app-build-manifest.json` on purpose. The manifest
lists a route's own entry chunks and is a poor proxy for the download: it
attributes `layout` and `template` to separate rows, so a client component in
the layout -- which every route pays for -- reads as costing nothing. The
`<script src>` list in the emitted HTML has no such gap. It is the download.

Three rules, in descending order of how much they matter:

1. **three.js must not be referenced by any page's HTML.** This is the whole
   point of the `next/dynamic({ ssr: false })` boundary in `components/three/`.
   If somebody imports `HeartScene` directly from a page, the build still
   succeeds, the page still works on their machine, and every visitor to every
   route silently starts paying several hundred kilobytes. Nothing else catches
   that.
2. **The shared set stays small.** It is paid by every route including the ones
   with no chart, no model and no animation on them.
3. **No single route blows the ceiling**, measured for the machine this project
   is marked on: integrated graphics, no discrete GPU.

Run after `next build`; wired into the frontend build as `postbuild`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
OUT_DIR = FRONTEND / "out"

#: Gzipped kilobytes every route pays, whatever is on it.
SHARED_BUDGET_KB = 130.0

#: Gzipped kilobytes any one route may cost on first load.
ROUTE_BUDGET_KB = 260.0

#: Strings that appear in a library's **body**, not merely in an import
#: specifier. `ScrollTrigger` would be the obvious marker for GSAP and is the
#: wrong one: a dynamic `import('gsap/ScrollTrigger')` leaves that name in the
#: importing chunk as a module id, so the marker matches the chunk that
#: correctly defers the library. `scrollerProxy` is a method defined inside
#: ScrollTrigger itself and appears only where the library really is.
LAZY_MARKERS = {
    "three.js": "WebGLRenderer",
    "gsap ScrollTrigger": "scrollerProxy",
    # ECharts is ~300 kB. `zrender` is its rendering engine and appears only
    # where the library itself is bundled.
    "echarts": "zrender",
    # WaveSurfer's exported class name. Present in its own chunk, nowhere else.
    "wavesurfer.js": "WaveSurfer",
}

#: Libraries that must not reach the browser AT ALL, with the evidence that
#: they nevertheless ran.
#:
#: KaTeX is a build-time dependency: `Equations.tsx` renders to HTML during the
#: static export. If a page marks itself `'use client'` and imports it, the
#: library follows across the boundary and 74 kB lands in the browser -- which
#: is exactly what happened the first time /design was written, and what the
#: server/client split of that page exists to prevent.
#:
#: "absent from every chunk" alone would also pass if the equations stopped
#: rendering entirely, so each entry names a string its OUTPUT leaves in the
#: exported HTML. Both halves have to hold.
SERVER_ONLY: dict[str, dict[str, str]] = {
    "katex": {
        "chunk_marker": "katex",
        "html_marker": "katex-html",
        "why": (
            "Equations render at build time so the page ships finished HTML, the "
            "library never reaches the browser, and a malformed formula fails the "
            "build rather than printing red text on a page nobody re-checks."
        ),
    },
}

#: One `<script>` tag, captured whole so its attributes can be inspected.
_SCRIPT_TAG = re.compile(r"<script[^>]*>", re.IGNORECASE)
_SRC_ATTR = re.compile(r'src="([^"]+)"')


def gzip_kb(path: Path) -> float:
    """Gzipped size in kB, which is what a browser actually downloads."""
    return len(gzip.compress(path.read_bytes(), 9)) / 1024.0


def _asset_path(src: str) -> Path | None:
    """Map a `/_next/...` URL to its file in `out/`."""
    if not src.startswith("/_next/") or not src.endswith(".js"):
        return None
    candidate = OUT_DIR / "_next" / src[len("/_next/") :]
    return candidate if candidate.is_file() else None


def page_scripts() -> dict[str, set[str]]:
    """Every exported page, and the scripts its HTML tells the browser to load."""
    if not OUT_DIR.is_dir():
        raise SystemExit(
            "no export to measure: " + str(OUT_DIR) + " is missing. Run `npm run build` first."
        )
    pages: dict[str, set[str]] = {}
    for html in sorted(OUT_DIR.rglob("*.html")):
        route = "/" + html.parent.relative_to(OUT_DIR).as_posix().strip(".")
        if html.name != "index.html":
            route = "/" + html.relative_to(OUT_DIR).as_posix()
        route = route.replace("//", "/")
        text = html.read_text(encoding="utf-8", errors="ignore")
        pages[route] = {
            src
            for tag in _SCRIPT_TAG.findall(text)
            # `nomodule` is the legacy-browser polyfill bundle. Every browser
            # that supports ES modules -- which is every browser this dashboard
            # targets -- skips it entirely, so counting its 38 kB would inflate
            # the measurement by more than the whole animation layer costs.
            if "nomodule" not in tag.lower()
            for src in _SRC_ATTR.findall(tag)
            if _asset_path(src) is not None
        }
    if not pages:
        raise SystemExit("no HTML found under " + str(OUT_DIR))
    return pages


def files_containing(marker: str) -> set[str]:
    """Exported JS files whose text contains ``marker``, as `/_next/...` URLs."""
    found: set[str] = set()
    root = OUT_DIR / "_next" / "static"
    if not root.is_dir():
        return found
    for path in root.rglob("*.js"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if marker in text:
            found.add("/_next/" + path.relative_to(OUT_DIR / "_next").as_posix())
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-budget-kb", type=float, default=SHARED_BUDGET_KB)
    parser.add_argument("--route-budget-kb", type=float, default=ROUTE_BUDGET_KB)
    parser.add_argument("--json", action="store_true", help="emit the measurement as JSON")
    args = parser.parse_args(argv)

    pages = page_scripts()

    sizes: dict[str, float] = {}

    def size_of(src: str) -> float:
        if src not in sizes:
            path = _asset_path(src)
            sizes[src] = 0.0 if path is None else gzip_kb(path)
        return sizes[src]

    shared: set[str] = set.intersection(*pages.values()) if pages else set()
    shared_kb = sum(size_of(src) for src in shared)

    routes = [
        {
            "route": route,
            "first_load_kb": round(sum(size_of(src) for src in scripts), 1),
            "n_scripts": len(scripts),
        }
        for route, scripts in sorted(pages.items())
    ]

    failures: list[str] = []

    referenced: set[str] = set()
    for scripts in pages.values():
        referenced.update(scripts)

    lazy_report: dict[str, dict[str, object]] = {}
    for label, marker in LAZY_MARKERS.items():
        carriers = files_containing(marker)
        leaked = sorted(carriers & referenced)
        lazy_report[label] = {"chunks_containing": len(carriers), "in_first_load": leaked}
        if leaked:
            failures.append(
                label
                + " is fetched on first load by at least one page ("
                + ", ".join(leaked)
                + "). It must stay behind a dynamic import; a direct import from a "
                "page puts it on every route that shares a chunk with that page."
            )
        elif not carriers:
            failures.append(
                label
                + " was not found in any exported chunk (marker "
                + marker
                + "). Either the library is no longer bundled -- in which case this "
                "check is measuring nothing -- or minification renamed the marker."
            )

    html_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in OUT_DIR.rglob("*.html")
    )
    server_only_report: dict[str, dict[str, object]] = {}
    for label, rule in SERVER_ONLY.items():
        bundled = sorted(files_containing(rule["chunk_marker"]))
        rendered = rule["html_marker"] in html_text
        server_only_report[label] = {"chunks_containing": bundled, "rendered": rendered}
        if bundled:
            failures.append(
                label
                + " reached the browser: it is bundled in "
                + ", ".join(bundled)
                + ". "
                + rule["why"]
                + " A page that imports it must not be a client component."
            )
        if not rendered:
            failures.append(
                label
                + " left no output in the exported HTML (looked for "
                + rule["html_marker"]
                + "). It is absent from the client bundle, but it does not appear "
                "to have run at build time either."
            )

    if shared_kb > args.shared_budget_kb:
        failures.append(
            "the shared script set is "
            + format(shared_kb, ".1f")
            + " kB gzipped, over the "
            + format(args.shared_budget_kb, ".1f")
            + " kB budget. Every route pays this, including the ones with nothing on them."
        )

    for entry in routes:
        size = float(entry["first_load_kb"])  # type: ignore[arg-type]
        if size > args.route_budget_kb:
            failures.append(
                str(entry["route"])
                + " first-load JS is "
                + format(size, ".1f")
                + " kB gzipped, over the "
                + format(args.route_budget_kb, ".1f")
                + " kB budget"
            )

    measurement = {
        "measured_from": "frontend/out (exported HTML script tags, gzipped)",
        "shared_kb": round(shared_kb, 1),
        "shared_budget_kb": args.shared_budget_kb,
        "route_budget_kb": args.route_budget_kb,
        "n_pages": len(pages),
        "routes": routes,
        "lazy": lazy_report,
        "server_only": server_only_report,
        "failures": failures,
    }

    if args.json:
        print(json.dumps(measurement, indent=2))
    else:
        print("bundle budget -- gzipped, from the scripts frontend/out/ actually loads")
        print(
            "  shared by all "
            + str(len(pages))
            + " pages: "
            + format(shared_kb, ".1f")
            + " kB  (budget "
            + format(args.shared_budget_kb, ".1f")
            + ")"
        )
        for entry in sorted(routes, key=lambda e: -float(e["first_load_kb"])):  # type: ignore[arg-type]
            print(
                "  "
                + str(entry["route"]).ljust(30)
                + format(float(entry["first_load_kb"]), "7.1f")  # type: ignore[arg-type]
                + " kB"
            )
        for label, report in server_only_report.items():
            print(
                "  server-only: "
                + label.ljust(13)
                + ("in the client bundle" if report["chunks_containing"] else "not in any chunk")
                + ", "
                + ("rendered into the HTML" if report["rendered"] else "NO OUTPUT IN HTML")
            )
        for label, report in lazy_report.items():
            state = "LEAKED into first load" if report["in_first_load"] else "not in first load"
            print(
                "  lazy: "
                + label.ljust(20)
                + str(report["chunks_containing"])
                + " chunk(s), "
                + state
            )

    if failures:
        print("")
        for failure in failures:
            print("FAIL: " + failure)
        return 1

    print("")
    print("within budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
