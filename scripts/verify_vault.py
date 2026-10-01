"""Check the Obsidian vault graph: no dead [[links]], valid frontmatter on graph notes, no orphans.

    python scripts/verify_vault.py [vault]
"""

import re
import sys
from datetime import date
from pathlib import Path

import yaml

GRAPH_DIRS = {"facts": "fact", "hypotheses": "hypothesis", "experiments": "experiment", "decisions": "decision", "findings": "finding"}
REQUIRED = ("type", "id", "created", "status", "links")
STATUSES = {"open", "testing", "confirmed", "rejected", "superseded"}
LINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def main(root: Path) -> int:
    notes = {p.stem: p for p in root.rglob("*.md") if ".obsidian" not in p.parts and ".trash" not in p.parts}
    errors, incoming = [], {k: 0 for k in notes}
    for stem, p in notes.items():
        text = p.read_text(encoding="utf-8")
        for target in LINK.findall(strip_code(text)):
            target = target.strip().split("/")[-1]
            if target not in notes:
                errors.append(f"dead link [[{target}]] in {p.relative_to(root)}")
            elif target != stem:
                incoming[target] += 1
        typ = GRAPH_DIRS.get(p.parent.name)
        if typ is None:
            continue
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if not m:
            errors.append(f"no frontmatter {p.relative_to(root)}")
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        errors += [f"missing {k} in {p.relative_to(root)}" for k in REQUIRED if fm.get(k) in (None, "")]
        if fm.get("type") != typ:
            errors.append(f"wrong type {fm.get('type')!r} in {p.relative_to(root)}")
        if fm.get("status") not in STATUSES:
            errors.append(f"bad status {fm.get('status')!r} in {p.relative_to(root)}")
        if not isinstance(fm.get("created"), date):
            errors.append(f"bad created in {p.relative_to(root)}")
        if not isinstance(fm.get("links"), list):
            errors.append(f"links not a list in {p.relative_to(root)}")
        if "## Links" not in text:
            errors.append(f"no '## Links' section in {p.relative_to(root)}")
    errors += [f"orphan (no incoming links) {notes[k].relative_to(root)}" for k, n in incoming.items() if n == 0 and k != "00_INDEX"]
    print(f"notes: {len(notes)}")
    print("\n".join(errors) if errors else "ALL CHECKS PASSED")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "vault")))
