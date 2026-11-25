# Lists the number macros whose values changed between two numbers.tex files (old, new).
import re, sys
old_p, new_p = sys.argv[1], sys.argv[2]
rx = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}$")
def load(p): return dict(rx.match(l).groups() for l in open(p) if rx.match(l))
o, n = load(old_p), load(new_p)
changed = [(k, o[k], n[k]) for k in n if k in o and o[k] != n[k]]
added = [k for k in n if k not in o]; removed = [k for k in o if k not in n]
print(f"{len(n)} macros; {len(changed)} changed, {len(added)} added, {len(removed)} removed\n")
w = max((len(k) for k, _, _ in changed), default=10)
for k, a, b in changed: print(f"{k:{w}s}  {a[:40]:40s} -> {b[:40]}")
if added: print("\nadded:", ", ".join(added))
if removed: print("\nremoved:", ", ".join(removed))
