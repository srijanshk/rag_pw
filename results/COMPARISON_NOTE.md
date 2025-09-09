# RAG COT Comparative Results

This note summarizes EM accuracy for recent runs across modes: Zero COT, Static-RAG-COT, and Dynamic-RAG-COT (by corpus).


## Dataset: gsm8k

- Zero COT: 82.11% (1083/1319) — `rag_pw/results/cot/gsm_COT_20250814_175905.jsonl`
- Static-RAG-COT: 75.82% (1000/1319) — `rag_pw/results/static/gsm_STATIC_COT_20250816_102840.jsonl`
- Dynamic-RAG-COT (mathpile, summary): 73.54% (970/1319), Retrieval Exec: 1.36% — with: 11.11% (2/18), without: 74.4% (968/1301) — `rag_pw/results/dynamic/gsm8k_mathpile_summary_20250906_0646_PROMPT2.0.json`
- Dynamic-RAG-COT (mathpile, raw): 74.37% (981/1319), Retrieval Exec: 0.83% — with: 18.18% (2/11), without: 74.85% (979/1308) — `rag_pw/results/dynamic/gsm8k_mathpile_raw_20250908_0831_PROMPT2.0.json`
- Dynamic-RAG-COT (openmath, summary): 75.59% (997/1319), Retrieval Exec: 0.61% — with: 37.5% (3/8), without: 75.82% (994/1311) — `rag_pw/results/dynamic/gsm8k_openmath_summary_20250908_0831_PROMPT2.0.json`
- Dynamic-RAG-COT (openmath, raw): 74.53% (983/1319), Retrieval Exec: 1.36% — with: 22.22% (4/18), without: 75.25% (979/1301) — `rag_pw/results/dynamic/gsm8k_openmath_raw_20250908_0831_PROMPT2.0.json`


## Dataset: math500

- Zero COT: 44.0% (220/500) — `rag_pw/results/cot/math500_COT_20250905_124208.jsonl`
- Static-RAG-COT: 42.2% (211/500) — `rag_pw/results/static/math500_STATIC_COT_20250905_165844.jsonl`
- Dynamic-RAG-COT (mathpile, summary): 32.0% (160/500), Retrieval Exec: 9.8% — with: 14.29% (7/49), without: 33.92% (153/451) — Δ vs Zero -12.00pp; Δ vs Static -10.20pp — `rag_pw/results/dynamic/math500_mathpile_summary_20250906_0646_PROMPT2.0.json`
- Dynamic-RAG-COT (mathpile, raw): 30.8% (154/500), Retrieval Exec: 9.0% — with: 11.11% (5/45), without: 32.75% (149/455) — Δ vs Zero -13.20pp; Δ vs Static -11.40pp — `rag_pw/results/dynamic/math500_mathpile_raw_20250908_0831_PROMPT2.0.json`
- Dynamic-RAG-COT (openmath, summary): 33.0% (165/500), Retrieval Exec: 11.2% — with: 1.79% (1/56), without: 36.94% (164/444) — Δ vs Zero -11.00pp; Δ vs Static -9.20pp — `rag_pw/results/dynamic/math500_openmath_summary_20250908_0831_PROMPT2.0.json`
- Dynamic-RAG-COT (openmath, raw): 34.0% (170/500), Retrieval Exec: 13.0% — with: 15.38% (10/65), without: 36.78% (160/435) — Δ vs Zero -10.00pp; Δ vs Static -8.20pp — `rag_pw/results/dynamic/math500_openmath_raw_20250908_0831_PROMPT2.0.json`
