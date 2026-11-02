"""Downloadable reports: Markdown, JSON, CSV and (if fpdf2 is installed) PDF."""
from __future__ import annotations

import json
from datetime import datetime

from src.next_life_engine import cost_table, pathway_table
from src.utils import GRADE_LABELS, PATHWAY_HEADLINE, PATHWAY_LABELS, PATHWAYS, inr

DISCLAIMER = ("Synthetic, simulated estimates for decision support only. Not a financial guarantee and not a "
              "replacement for physical inspection. Transformation ideas are estimated/simulated and need human "
              "validation. Actual economics depend on market conditions.")


def _clean(label: str | None) -> str:
    return label or "Product"


def build_markdown(a: dict, label: str | None = None) -> str:
    r, c, k = a["record"], a["condition"], a["recommended"]
    best = a["pathways"][k]
    L = [f"# ReturnCycle AI — Recovery Analysis", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
         f"**Product:** {_clean(label or r.get('product_name'))} ({r['category']})",
         f"**Original value:** {inr(r['original_value'])}",
         f"**Condition:** score {c['score']:.0f}/100 · Grade {c['grade']} ({GRADE_LABELS[c['grade']]})", "",
         "## Possible recovery paths", "| Pathway | Gross | Costs | Net | Preserved | Days | Score |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key in PATHWAYS:
        p = a["pathways"][key]
        L.append(f"| {PATHWAY_LABELS[key]} | {inr(p['gross'])} | {inr(p['total_cost'])} | {inr(p['net'])} | "
                 f"{p['value_preserved']:.1f}% | {p['days']:.1f} | {p['score']:.0f} |")
    L += ["", "## Recommended path", f"**{PATHWAY_HEADLINE[k]}**", f"- Expected net recovery: {inr(best['net'])}",
          f"- Estimated costs: {inr(best['total_cost'])}", f"- Value preserved: {best['value_preserved']:.1f}%",
          f"- Processing time: {best['days']:.1f} days", f"- Confidence: {a['confidence']:.0%} (heuristic)", "",
          "## Key factors", a.get("reasoning", ""), "", "### Condition explanation", "```", c["explanation"], "```", ""]
    if a.get("overrides"):
        L += ["## Simulation overrides applied", json.dumps(a["overrides"], indent=2), ""]
    L += ["## Disclaimer", DISCLAIMER]
    return "\n".join(L)


def build_json(a: dict, label: str | None = None) -> str:
    k = a["recommended"]
    out = {"product": _clean(label or a["record"].get("product_name")), "category": a["record"]["category"],
           "original_value": a["record"]["original_value"],
           "condition": {"score": a["condition"]["score"], "grade": a["condition"]["grade"],
                         "issues": a["condition"]["issues"]},
           "recommended_pathway": PATHWAY_LABELS[k], "confidence": a["confidence"],
           "pathways": {PATHWAY_LABELS[key]: {"gross": round(p["gross"], 2), "costs": {n: round(v, 2) for n, v in p["costs"].items()},
                                              "net": round(p["net"], 2), "value_preserved_pct": round(p["value_preserved"], 2),
                                              "days": round(p["days"], 1), "score": p["score"]}
                        for key, p in a["pathways"].items()},
           "reasoning": a.get("reasoning", ""), "overrides": a.get("overrides", {}), "disclaimer": DISCLAIMER}
    return json.dumps(out, indent=2)


def build_csv(a: dict) -> str:
    return pathway_table(a).to_csv(index=False)


def build_pdf(a: dict, label: str | None = None):
    """Returns PDF bytes, or None when fpdf2 is not installed."""
    try:
        from fpdf import FPDF
    except Exception:  # noqa: BLE001
        return None
    try:
        def t(s):  # core PDF fonts are latin-1 only
            return str(s).replace("₹", "Rs. ").replace("—", "-").replace("·", "|").replace("−", "-") \
                .encode("latin-1", "replace").decode("latin-1")
        r, c, k = a["record"], a["condition"], a["recommended"]
        best = a["pathways"][k]
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 9, "ReturnCycle AI - Recovery Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, t(f"Generated {datetime.now():%Y-%m-%d %H:%M}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        for line in (f"Product: {_clean(label or r.get('product_name'))} ({r['category']})",
                     f"Original value: {inr(r['original_value'])}",
                     f"Condition: {c['score']:.0f}/100 | Grade {c['grade']} ({GRADE_LABELS[c['grade']]})"):
            pdf.cell(0, 6, t(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Possible recovery paths", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 9)
        widths = [44, 28, 28, 28, 24, 18, 16]
        for w, h in zip(widths, ["Pathway", "Gross", "Costs", "Net", "Preserved", "Days", "Score"]):
            pdf.cell(w, 6, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for key in PATHWAYS:
            p = a["pathways"][key]
            vals = [PATHWAY_LABELS[key], inr(p["gross"]), inr(p["total_cost"]), inr(p["net"]),
                    f"{p['value_preserved']:.1f}%", f"{p['days']:.1f}", f"{p['score']:.0f}"]
            for w, v in zip(widths, vals):
                pdf.cell(w, 6, t(v), border=1)
            pdf.ln()
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, t(f"Recommended: {PATHWAY_HEADLINE[k]}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in (f"Expected net recovery: {inr(best['net'])}", f"Estimated costs: {inr(best['total_cost'])}",
                     f"Value preserved: {best['value_preserved']:.1f}%", f"Processing time: {best['days']:.1f} days",
                     f"Confidence: {a['confidence']:.0%} (heuristic)"):
            pdf.cell(0, 6, t(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Key factors", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, t(a.get("reasoning", "")), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4.5, t("Disclaimer: " + DISCLAIMER), new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())
    except Exception:  # noqa: BLE001
        return None
