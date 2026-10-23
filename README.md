# 🛡️ ML-Based Network Intrusion Detection System (IDS)

> **IBM Internship Project — Cybersecurity Track**

A real-time Network Intrusion Detection System that uses Machine Learning (Random Forest) to classify network traffic as **Normal, DoS, Probe, R2L, or U2R**, with a live Streamlit dashboard — packaged as a **single file: `app.py`**.

---

## 📸 Features

- ✅ **ML-based detection** — Random Forest trained on NSL-KDD dataset
- ✅ **5-class classification** — Normal, DoS, Probe, R2L, U2R
- ✅ **Live packet capture** — Scapy-based real-time traffic sniffing
- ✅ **Alert management** — SQLite-backed persistent alert store
- ✅ **Interactive dashboard** — Streamlit + Plotly visualisations
- ✅ **Single-file design** — entire project in one `app.py`

---

## 🏗️ Architecture

```
Network Traffic → Packet Capture (Scapy)
                       ↓
               Feature Extraction (41 features)
                       ↓
           ML Model (Random Forest)
                       ↓
              ┌────────┴────────┐
           Normal           Attack 🚨
                               ↓
                        Alert Manager (SQLite)
                               ↓
                      Streamlit Dashboard
```

---

## 📁 Project Structure

```
ibm project/
├── app.py               ← Entire project (features + train + predict + capture + dashboard)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── data/
│   ├── raw/             ← Place KDDTrain+.txt and KDDTest+.txt here
│   └── processed/       ← Auto-generated preprocessed CSVs
└── models/              ← Saved model, scaler, encoders, metrics, plots
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the NSL-KDD dataset

```bash
# Option A — auto download (no account needed)
python -c "
import urllib.request, os
os.makedirs('data/raw', exist_ok=True)
urllib.request.urlretrieve('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt', 'data/raw/KDDTrain+.txt')
urllib.request.urlretrieve('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt',  'data/raw/KDDTest+.txt')
print('Done')
"

# Option B — manual download
# https://www.kaggle.com/datasets/hassan06/nslkdd
# Place KDDTrain+.txt and KDDTest+.txt inside data/raw/
```

### 3. Preprocess the data

```bash
python app.py --preprocess
```

### 4. Train the model

```bash
# Multiclass (DoS / Probe / R2L / U2R / normal)
python app.py --train

# Binary (normal vs attack)
python app.py --train --binary

# With 5-fold cross-validation
python app.py --train --cross-val
```

### 5. Run the dashboard

```bash
python -m streamlit run app.py
# Open http://localhost:8501
```

### 6. Start live capture (requires Administrator / root)

```bash
# Windows (run terminal as Administrator)
python app.py --capture --iface "Wi-Fi"

# Linux / macOS
sudo python app.py --capture --iface eth0
```

---

## 🖥️ Dashboard Sections

| Section | Description |
|---|---|
| KPI Cards | Total connections, attacks detected, critical alerts, normal traffic |
| Attack Distribution | Donut chart of attack type breakdown |
| Attack Timeline | 10-minute resampled area chart of attack frequency |
| Severity Breakdown | Bar chart — CRITICAL / HIGH / MEDIUM / LOW |
| Recent Alerts | Colour-highlighted table of the 20 most recent attacks |
| Model Performance | Accuracy, F1, per-class report, confusion matrix, feature importance |
| Top Attacking IPs | Horizontal bar chart of top 10 source IPs |
| Demo / Live Mode | Toggle between simulated and real DB data |

---

## ⚙️ CLI Reference

```
usage: app.py [-h] [--preprocess] [--train] [--capture]
              [--binary] [--cross-val] [--n-estimators N]
              [--iface IFACE] [--count COUNT]

options:
  --preprocess          Preprocess raw NSL-KDD data → data/processed/
  --train               Train the Random Forest model
  --capture             Start live packet capture
  --binary              Binary mode  (normal vs attack)
  --cross-val           Run 5-fold cross-validation before training
  --n-estimators N      Number of trees  (default: 100)
  --iface IFACE         Network interface for capture  (default: eth0)
  --count COUNT         Packets to capture, 0 = infinite  (default: 0)
```

---

## 📊 Model Performance (NSL-KDD Test Set)

| Metric | Score |
|---|---|
| Accuracy | 75.2% |
| F1 Score (weighted) | 71.1% |
| Precision (weighted) | 81.7% |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| DoS | 96.1% | 76.9% | 85.4% |
| Probe | 87.3% | 67.5% | 76.2% |
| R2L | 98.5% | 4.8% | 9.2% |
| U2R | 60.0% | 1.5% | 2.9% |
| Normal | 64.9% | 97.4% | 77.9% |

> **Note:** Low recall on R2L and U2R is due to severe class imbalance in the NSL-KDD dataset (52 U2R training samples vs 67,343 Normal). Apply SMOTE or `class_weight="balanced"` to improve these classes.

---

## 🔍 Attack Categories

| Category | Examples |
|---|---|
| DoS | Neptune, Smurf, Back, Teardrop, Pod |
| Probe | Portsweep, Nmap, Ipsweep, Satan |
| R2L | FTP brute force, Guess_passwd, IMAP |
| U2R | Buffer overflow, Rootkit, Perl |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML | scikit-learn (Random Forest) |
| Packet Capture | Scapy |
| Dashboard | Streamlit + Plotly |
| Database | SQLite 3 |
| Dataset | NSL-KDD |
| Visualisation | matplotlib + seaborn |

---

## 📄 License

MIT License — free to use for educational and internship purposes.

---

*Built as part of IBM Internship — Cybersecurity Track*
