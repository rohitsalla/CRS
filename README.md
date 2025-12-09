#  **LLM Reliability Benchmark (CRS)**

**Official implementation for the AAAI 2026 paper:**

### **Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models**

This repository provides the reference implementation of the **Composite Reliability Score (CRS)**, a unified benchmark that evaluates **Calibration**, **Robustness**, and **Uncertainty** for modern Large Language Models.

CRS reveals failure modes that accuracy alone cannot capture, helping researchers and practitioners assess model reliability under realistic conditions.

---

## 📁 Project Structure

```
├── src/crs/                 # Core CRS implementation
│   ├── calibration/
│   ├── robustness/
│   ├── uncertainty/
│   ├── utils/
│   └── crs_score.py
│
├── experiments/             # Scripts to run CRS components
│   ├── run_crs_pipeline.py
│   ├── run_robustness.py
│   ├── run_uncertainty.py
│   └── run_calibration.py
│
├── data/                    # Dataset directory (populated after download)
│   └── download.sh
│
├── figures/                 # Output plots and summaries
└── README.md
```

---

## Dataset Download

Datasets cannot be included directly due to licensing restrictions.
Use the provided script to download the required QA datasets (TriviaQA, NQ, SQuAD2.0, ARC, MedQA):

```bash
chmod +x data/download.sh
./data/download.sh
```

All datasets will be placed automatically in:

```
data/<dataset_name>/
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full CRS pipeline

```bash
python experiments/run_crs_pipeline.py
```

This computes:

* Calibration metrics (ECE, Brier Score, NLL)
* Robustness under noise, paraphrasing, and adversarial edits
* Uncertainty AUROC via MC Dropout + Ensembles
* Normalized CRS score

---

## Individual Evaluation Scripts

Run robustness evaluation:

```bash
python experiments/run_robustness.py
```

Run uncertainty evaluation:

```bash
python experiments/run_uncertainty.py
```

Run calibration evaluation:

```bash
python experiments/run_calibration.py
```

---

## Citation

If you use CRS in your research, please cite:

```
Salla, R.K., Saravanan, M., & Kota, S.R.R. (2026).
Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models.
AAAI Conference on Artificial Intelligence (AAAI 2026).
```

---

## Contributing

We welcome contributions that extend CRS to new tasks, models, or reliability dimensions.

Submit a pull request or open an issue with suggestions.

---

## 📬 Contact

For questions about CRS or the paper:

**Rohit Salla**
Virginia Tech, ECE Department
📧 [rohits25@vt.edu](mailto:rohits25@vt.edu)

# CRS
