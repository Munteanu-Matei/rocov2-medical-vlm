"""Score enriched/AUDIT_SHEET.csv after manual grading.

The sheet over-samples risky strata (20 each), so a raw mean would be biased
low.  We weight each stratum by its true share of the matched population and
report a Wilson interval, which is correct for proportions near 0 or 1 where
the normal approximation fails.
"""
import pandas as pd, math, sys

# true stratum shares, measured on 2500 test captions -> 4106 matched pairs
POP = {"rare": 3093, "frequent": 469, "negation-context": 309,
       "multi-token": 118, "short/abbrev": 117}
N = sum(POP.values())

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

df = pd.read_csv("enriched/AUDIT_SHEET.csv")
g = df[df.correct_YN.astype(str).str.upper().isin(["Y","N"])].copy()
if len(g) == 0:
    sys.exit("no rows graded yet -- fill the correct_YN column with Y or N")
g["ok"] = g.correct_YN.astype(str).str.upper().eq("Y")

print(f"graded {len(g)}/{len(df)} rows\n")
print(f"  {'stratum':18s} {'graded':>7s} {'correct':>8s} {'prec':>7s}  {'weight':>7s}")
print("  " + "-"*56)
num = den = 0.0
for s, sub in g.groupby("stratum"):
    w = POP.get(s, 0) / N
    p = sub.ok.mean()
    num += w * p; den += w
    print(f"  {s:18s} {len(sub):7d} {int(sub.ok.sum()):8d} {p:6.1%}  {w:6.1%}")

overall = num / den
k, n = int(g.ok.sum()), len(g)
lo, hi = wilson(k, n)
print(f"\n  weighted precision : {overall:.1%}")
print(f"  unweighted         : {k/n:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
print(f"\n  -> of {16597} test labels added, an estimated {int(16597*overall):,} are correct")
print(f"     and {int(16597*(1-overall)):,} are spurious.")
print("\n  NB: the CI is on the unweighted sample. With 20/stratum the per-stratum")
print("      intervals are wide -- treat this as an order-of-magnitude estimate,")
print("      and grade more rows in any stratum that looks bad.")
