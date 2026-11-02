# ReturnCycle AI — Returned Product Next-Life & Value Recovery Intelligence

> **A returned product is not necessarily a lost product.**
> ReturnCycle AI evaluates what the product can become next.

*This project is an independent synthetic-data prototype inspired by challenges in e-commerce returns, recommerce and circular recovery.*

## 1. Problem
Returned or damaged products are often treated as a loss and sent to disposal or low-value recycling (`DAMAGED = RECYCLE`), even when significant value remains inside them.

## 2. Why it matters
Returns erode margin, create waste and add reverse-logistics cost. Choosing the right next life for each item recovers money and keeps products in use longer.

## 3. Solution
For every return the system answers: **"What is the highest-value practical next life for this product?"**

`RETURNED → INSPECTED → CONDITION GRADED → VALUE ESTIMATED → PATHWAYS GENERATED → BEST NEXT LIFE`

Six pathways are costed and ranked: **Resale · Refurbish · Component Recovery · Transformation/Upcycling · Material Recovery · Disposal**.

## 4. Key innovation
* Value-first reasoning instead of `damaged → recycle`.
* A **Transformation engine** ("Could this become something else?") — rule-based, clearly labelled *estimated / simulated*.
* A **counterfactual simulator** that recalculates every pathway (and the recommendation) when any assumption changes.

## 5. Architecture
```
Synthetic Data → Data Cleaning → Feature Engineering → Condition Intelligence
→ Recovery Value Estimation → Next-Life Candidates → Economic Optimization
→ Counterfactual Simulation → Streamlit Decision Dashboard
```
| Component | Technique |
|---|---|
| Condition score/grade | Deterministic weighted rules |
| Net recovery per pathway | Deterministic arithmetic |
| Ranking / recommendation | Normalised scoring optimizer |
| Transformation ideas | Rule-based catalogue |
| Fast grade / pathway / value prediction | XGBoost (fallback: RandomForest) |
| Explanations | Templated reasoning + SHAP (falls back to feature importance) |

## 6. Tech stack
Python · Streamlit · Plotly · Pandas/NumPy · scikit-learn · XGBoost · SHAP · PyYAML · fpdf2 · pytest.

## 7. Dataset
**Synthetic demonstration dataset created for ReturnCycle AI.** No Amazon/Flipkart/private data.
`demo`: 25,000 orders · 6,000 customers · 800 products · ~9,000 returns. `full`: 100,000 orders · 20,000 customers · 2,000 products · ~36,000 returns.
The demo is *return-enriched* (return rates above real-world averages) so that 25k orders yield 8–12k returns. 12 categories with different economics. "Current practice" outcomes are a **simulated legacy policy**.

## 8. ML methodology
Three models trained on an 80/20 split (metrics in `models/metrics.json`, shown on *Model Insights*): condition-grade classifier, recovery-pathway classifier, recovery-value regressor. They learn to reproduce the synthetic engine for fast batch scoring — the deterministic engine remains authoritative. Metrics are computed, never fabricated.

## 9. Recovery optimization
`net = gross − repair − refurbishment − processing − transformation − logistics − warehouse − disposal − expected_loss`; `value preserved % = net / original value × 100`.
`score = economic + suitability + usability + market − cost − time − risk` (weights in `config.yaml`, editable in the app), normalised to 0-100. Disposal is only eligible when all alternatives lose more value than disposal.

## 10. Transformation engine
Per-category rules (e.g. laptop → NAS kit / spare-parts bundle; furniture → storage unit; clothing → upcycled garment). Needs human validation.

## 11. Screenshots
`docs/screenshots/dashboard.png`, `inspection.png`, `simulator.png` *(placeholders — add your own)*

## 12. Installation
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 13. Running
```bash
python src/data_generator.py --mode demo     # optional: sample data is already bundled
python src/model_training.py                 # optional: pre-trained models are bundled
streamlit run app.py
```
(`--mode full` generates the larger dataset.) If data or models are missing the app tells you what to run and offers in-app buttons (About → Data tools, Model Insights → Train).

## 14. Testing
```bash
pytest
```

## 15. Limitations
Synthetic data; simulated economics; heuristic confidence; per-category assumptions are illustrative; ML metrics reflect agreement with the synthetic engine, not real-world accuracy. Transformation and Disposal rarely win under default assumptions (by design), but can be made to win in the simulator.

## 16. Responsible AI
Decision support, not decision replacement: physical inspection is still required, transformation ideas need human validation, all values are simulated estimates, and outputs are probabilistic. See the in-app *Methodology & Responsible AI* page.

## 17. Future improvements
Real inspection images (CV condition grading), live market-price feeds, per-seller/warehouse constraints, multi-item batch optimisation, uncertainty intervals, A/B tests of recommendations.
