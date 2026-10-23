"""
app.py
------
ML-Based Network Intrusion Detection System — single-file edition.
Combines feature engineering, model training, inference, alert management,
live packet capture, and the Streamlit dashboard in one file.

Run dashboard:
    python -m streamlit run app.py

Train model (CLI):
    python app.py --train

Preprocess data (CLI):
    python app.py --preprocess

Live capture (CLI, requires Admin/root):
    python app.py --capture --iface "Wi-Fi"
"""

# ══════════════════════════════════════════════════════════════════════════════
#  Standard library
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import json
import time
import random
import sqlite3
import threading
import argparse
import numpy as np
from datetime import datetime
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable

# ══════════════════════════════════════════════════════════════════════════════
#  Third-party
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.model_selection import cross_val_score

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  Paths & constants
# ══════════════════════════════════════════════════════════════════════════════
DATA_RAW_DIR   = "data/raw"
DATA_PROC_DIR  = "data/processed"
MODEL_DIR      = "models"
DB_PATH        = "data/alerts.db"
METRICS_PATH   = "models/metrics.json"
CM_PATH        = "models/confusion_matrix.png"
FI_PATH        = "models/feature_importance.png"
TRAIN_FILE     = "data/raw/KDDTrain+.txt"
TEST_FILE      = "data/raw/KDDTest+.txt"

ATTACK_COLORS = {
    "normal": "#00c853",
    "DoS":    "#ff4b4b",
    "Probe":  "#ffd600",
    "R2L":    "#e040fb",
    "U2R":    "#40c4ff",
    "attack": "#ff4b4b",
}

# ══════════════════════════════════════════════════════════════════════════════
#  §1  FEATURES  (preprocessing pipeline)
# ══════════════════════════════════════════════════════════════════════════════
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

ATTACK_MAP = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df.drop(columns=["difficulty"], inplace=True)
    return df


def map_labels(df: pd.DataFrame, binary: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].str.strip().str.lower().map(
        lambda x: ATTACK_MAP.get(x, "unknown")
    )
    if binary:
        df["label"] = df["label"].apply(
            lambda x: "normal" if x == "normal" else "attack"
        )
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )
    return df, encoders


def scale_features(X: np.ndarray, scaler=None, fit: bool = True):
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def preprocess(
    train_path: str = TRAIN_FILE,
    test_path: str  = TEST_FILE,
    binary: bool    = False,
    out_dir: str    = DATA_PROC_DIR,
    model_dir: str  = MODEL_DIR,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Loading datasets...")
    train_df = load_dataset(train_path)
    test_df  = load_dataset(test_path)

    print("Mapping labels...")
    train_df = map_labels(train_df, binary=binary)
    test_df  = map_labels(test_df,  binary=binary)

    train_df = train_df[train_df["label"] != "unknown"].reset_index(drop=True)
    test_df  = test_df[test_df["label"]   != "unknown"].reset_index(drop=True)

    print("Encoding categoricals...")
    train_df, encoders = encode_categoricals(train_df, fit=True)
    test_df,  _        = encode_categoricals(test_df, encoders=encoders, fit=False)

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values

    print("Scaling features...")
    X_train_s, scaler = scale_features(X_train, fit=True)
    X_test_s,  _      = scale_features(X_test, scaler=scaler, fit=False)

    suffix    = "binary" if binary else "multiclass"
    train_out = pd.DataFrame(X_train_s, columns=feature_cols)
    train_out["label"] = y_train
    test_out  = pd.DataFrame(X_test_s, columns=feature_cols)
    test_out["label"]  = y_test

    train_out.to_csv(f"{out_dir}/train_{suffix}.csv", index=False)
    test_out.to_csv( f"{out_dir}/test_{suffix}.csv",  index=False)

    joblib.dump(encoders, f"{model_dir}/encoders.pkl")
    joblib.dump(scaler,   f"{model_dir}/scaler.pkl")
    print(f"Done. Train={len(train_out)} Test={len(test_out)}")
    print(train_out["label"].value_counts())
    return X_train_s, y_train, X_test_s, y_test, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  §2  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def load_processed(data_dir: str = DATA_PROC_DIR, binary: bool = False):
    suffix = "binary" if binary else "multiclass"
    train  = pd.read_csv(f"{data_dir}/train_{suffix}.csv")
    test   = pd.read_csv(f"{data_dir}/test_{suffix}.csv")
    fcols  = [c for c in train.columns if c != "label"]
    return (
        train[fcols].values, train["label"].values,
        test[fcols].values,  test["label"].values,
        fcols,
    )


def train_model(X_train, y_train, n_estimators: int = 100):
    print(f"Training Random Forest ({n_estimators} trees)...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test, feature_cols, out_dir: str = MODEL_DIR):
    os.makedirs(out_dir, exist_ok=True)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%")
    print(classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1, "report": report}, f, indent=2)

    # Confusion matrix
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance
    top_n   = 20
    idx     = np.argsort(clf.feature_importances_)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=clf.feature_importances_[idx],
                y=[feature_cols[i] for i in idx], palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=150)
    plt.close()
    return acc, f1


def run_training(binary: bool = False, n_estimators: int = 100,
                 cross_val: bool = False):
    X_train, y_train, X_test, y_test, fcols = load_processed(binary=binary)
    print(f"Train {X_train.shape} | Test {X_test.shape}")

    if cross_val:
        print("Running 5-fold CV...")
        clf_cv = RandomForestClassifier(n_estimators=n_estimators,
                                        random_state=42, n_jobs=-1)
        scores = cross_val_score(clf_cv, X_train, y_train, cv=5)
        print(f"CV Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    clf = train_model(X_train, y_train, n_estimators)
    evaluate_model(clf, X_test, y_test, fcols)

    suffix = "binary" if binary else "multiclass"
    joblib.dump(clf,   f"{MODEL_DIR}/ids_model_{suffix}.pkl")
    joblib.dump(fcols, f"{MODEL_DIR}/feature_cols.pkl")
    print(f"Model saved → {MODEL_DIR}/ids_model_{suffix}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
#  §3  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PredictionResult:
    label:         str
    confidence:    float
    probabilities: dict  = field(default_factory=dict)
    timestamp:     float = field(default_factory=time.time)
    is_attack:     bool  = False

    def __post_init__(self):
        self.is_attack = self.label != "normal"


class IDSPredictor:
    def __init__(self, model_dir: str = MODEL_DIR, binary: bool = False):
        suffix = "binary" if binary else "multiclass"
        mp = os.path.join(model_dir, f"ids_model_{suffix}.pkl")
        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"Model not found: {mp}\nRun:  python app.py --train"
            )
        self.model        = joblib.load(mp)
        self.encoders     = joblib.load(os.path.join(model_dir, "encoders.pkl"))
        self.scaler       = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
        self.classes      = self.model.classes_

    def _preprocess(self, record: dict) -> np.ndarray:
        row = {col: record.get(col, 0) for col in self.feature_cols}
        for col in CATEGORICAL_COLS:
            le  = self.encoders.get(col)
            val = str(row.get(col, ""))
            if le is not None:
                row[col] = le.transform([val])[0] if val in le.classes_ else -1
        X = np.array([[row[c] for c in self.feature_cols]], dtype=float)
        return self.scaler.transform(X)

    def predict_one(self, record: dict) -> PredictionResult:
        X     = self._preprocess(record)
        label = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return PredictionResult(
            label=label,
            confidence=float(np.max(proba)),
            probabilities={c: float(p) for c, p in zip(self.classes, proba)},
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = [self.predict_one(r.to_dict()) for _, r in df.iterrows()]
        return pd.concat([
            df.reset_index(drop=True),
            pd.DataFrame([{"prediction": r.label,
                           "confidence": r.confidence,
                           "is_attack":  r.is_attack} for r in results]),
        ], axis=1)

    def predict_csv(self, csv_path: str, out_path: Optional[str] = None):
        df = pd.read_csv(csv_path)
        result_df = self.predict_batch(df)
        if out_path:
            result_df.to_csv(out_path, index=False)
            print(f"Saved → {out_path}")
        attacks = result_df[result_df["is_attack"]]
        print(f"Total: {len(result_df)}  Attacks: {len(attacks)}")
        print(attacks["prediction"].value_counts())
        return result_df


# ══════════════════════════════════════════════════════════════════════════════
#  §4  ALERTS
# ══════════════════════════════════════════════════════════════════════════════
SEVERITY_MAP = {
    "DoS":    "HIGH",
    "U2R":    "CRITICAL",
    "R2L":    "HIGH",
    "Probe":  "MEDIUM",
    "attack": "HIGH",
}


@dataclass
class Alert:
    id:           int
    timestamp:    float
    src_ip:       str
    dst_ip:       str
    dst_port:     int
    attack_type:  str
    confidence:   float
    severity:     str  = "MEDIUM"
    acknowledged: bool = False

    @property
    def time_str(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


class AlertManager:
    def __init__(self, db_path: str = DB_PATH, persist: bool = True):
        self._lock    = threading.Lock()
        self._alerts: List[Alert] = []
        self._counter = 0
        self.persist  = persist
        if persist:
            os.makedirs("data", exist_ok=True)
            self.db_path = db_path
            self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY, timestamp REAL,
                src_ip TEXT, dst_ip TEXT, dst_port INTEGER,
                attack_type TEXT, confidence REAL,
                severity TEXT, acknowledged INTEGER DEFAULT 0)""")
            c.commit()

    def _save(self, a: Alert):
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?)",
                (a.id, a.timestamp, a.src_ip, a.dst_ip, a.dst_port,
                 a.attack_type, a.confidence, a.severity, int(a.acknowledged)),
            )
            c.commit()

    def add_alert(self, src_ip, dst_ip, dst_port, attack_type, confidence) -> Alert:
        with self._lock:
            self._counter += 1
            a = Alert(
                id=self._counter, timestamp=time.time(),
                src_ip=src_ip, dst_ip=dst_ip, dst_port=dst_port,
                attack_type=attack_type, confidence=confidence,
                severity=SEVERITY_MAP.get(attack_type, "MEDIUM"),
            )
            self._alerts.append(a)
        if self.persist:
            self._save(a)
        return a

    def get_alerts(self, limit=None, attack_type=None) -> List[Alert]:
        with self._lock:
            alerts = list(self._alerts)
        if attack_type:
            alerts = [a for a in alerts if a.attack_type == attack_type]
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit] if limit else alerts


# ══════════════════════════════════════════════════════════════════════════════
#  §5  LIVE CAPTURE
# ══════════════════════════════════════════════════════════════════════════════
PROTO_MAP = {6: "tcp", 17: "udp", 1: "icmp"}
PORT_SERVICE_MAP = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 80: "http", 110: "pop_3", 143: "imap4",
    443: "https", 3306: "mysql", 8080: "http_8001",
}
TCP_FLAG_MAP = {
    0x002: "S0", 0x012: "S1", 0x010: "SF",
    0x004: "REJ", 0x014: "RSTO", 0x001: "SH",
}


class ConnectionTracker:
    WINDOW = 2.0

    def __init__(self):
        self._lock   = threading.Lock()
        self._recent = deque()

    def record(self, ts, src_ip, dst_ip, dst_port):
        with self._lock:
            self._recent.append((ts, src_ip, dst_ip, dst_port))
            cutoff = ts - self.WINDOW
            while self._recent and self._recent[0][0] < cutoff:
                self._recent.popleft()

    def get_stats(self, ts, dst_ip, dst_port):
        with self._lock:
            recent = [(t, si, di, dp) for t, si, di, dp in self._recent
                      if t >= ts - self.WINDOW]
        same_dst = [r for r in recent if r[2] == dst_ip]
        same_srv = [r for r in same_dst if r[3] == dst_port]
        diff_srv = [r for r in same_dst if r[3] != dst_port]
        n = len(same_dst)
        return {
            "count":         n,
            "srv_count":     len(same_srv),
            "same_srv_rate": len(same_srv) / n if n else 0.0,
            "diff_srv_rate": len(diff_srv) / n if n else 0.0,
        }


def extract_features(pkt) -> Optional[dict]:
    if not pkt.haslayer(IP):
        return None
    ip       = pkt[IP]
    ts       = time.time()
    proto    = PROTO_MAP.get(ip.proto, "other")
    dst_port = src_port = 0
    flag_str = "OTH"
    urgent   = 0

    if pkt.haslayer(TCP):
        tcp      = pkt[TCP]
        dst_port = tcp.dport
        src_port = tcp.sport
        flag_str = TCP_FLAG_MAP.get(int(tcp.flags), "OTH")
        urgent   = 1 if tcp.urgptr > 0 else 0
    elif pkt.haslayer(UDP):
        dst_port = pkt[UDP].dport
        src_port = pkt[UDP].sport

    service = PORT_SERVICE_MAP.get(dst_port, "other")
    hot     = len(pkt[Raw].load) if pkt.haslayer(Raw) else 0

    return {
        "duration": 0, "protocol_type": proto, "service": service,
        "flag": flag_str, "src_bytes": len(pkt), "dst_bytes": 0,
        "land": 1 if ip.src == ip.dst else 0,
        "wrong_fragment": 0, "urgent": urgent,
        "hot": min(hot // 100, 30), "num_failed_logins": 0,
        "logged_in": 1 if dst_port in (22, 23, 21) else 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
        "num_root": 0, "num_file_creations": 0, "num_shells": 0,
        "num_access_files": 0, "num_outbound_cmds": 0,
        "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1,
        "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0, "dst_host_count": 1,
        "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
        "_src_ip": ip.src, "_dst_ip": ip.dst,
        "_dst_port": dst_port, "_ts": ts,
    }


class LiveCapture:
    def __init__(self, iface: str, on_result: Callable,
                 alert_manager: Optional[AlertManager] = None,
                 predictor: Optional[IDSPredictor] = None):
        self.iface         = iface
        self.on_result     = on_result
        self.alert_manager = alert_manager
        self.predictor     = predictor
        self.tracker       = ConnectionTracker()
        self._stop         = threading.Event()

    def _handle(self, pkt):
        if self._stop.is_set():
            return
        features = extract_features(pkt)
        if not features:
            return
        ts, src_ip = features["_ts"], features["_src_ip"]
        dst_ip, dst_port = features["_dst_ip"], features["_dst_port"]
        self.tracker.record(ts, src_ip, dst_ip, dst_port)
        features.update(self.tracker.get_stats(ts, dst_ip, dst_port))
        if self.predictor:
            result = self.predictor.predict_one(features)
            if self.alert_manager and result.is_attack:
                self.alert_manager.add_alert(
                    src_ip, dst_ip, dst_port,
                    result.label, result.confidence,
                )
            self.on_result(result, features)

    def start(self, count: int = 0):
        if not SCAPY_AVAILABLE:
            print("[ERROR] Scapy unavailable."); return
        print(f"[*] Capturing on {self.iface} ...")
        sniff(iface=self.iface, prn=self._handle, store=False,
              count=count, stop_filter=lambda _: self._stop.is_set())

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
#  §6  DASHBOARD  (Streamlit — only executed when run via streamlit)
# ══════════════════════════════════════════════════════════════════════════════
def _run_dashboard():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(
        page_title="IDS Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .metric-card { background:#1e2130; border-radius:10px; padding:20px;
                       border-left:4px solid; margin-bottom:10px; }
        .metric-red    { border-color:#ff4b4b; }
        .metric-green  { border-color:#00c853; }
        .metric-yellow { border-color:#ffd600; }
        .metric-blue   { border-color:#2196f3; }
    </style>
    """, unsafe_allow_html=True)

    # ── helpers ───────────────────────────────────────────────────────────────
    def load_alerts_db() -> pd.DataFrame:
        if not os.path.exists(DB_PATH):
            return pd.DataFrame(columns=[
                "id","timestamp","src_ip","dst_ip","dst_port",
                "attack_type","confidence","severity","acknowledged",
            ])
        with sqlite3.connect(DB_PATH) as conn:
            try:
                df = pd.read_sql_query(
                    "SELECT * FROM alerts ORDER BY timestamp DESC", conn)
            except Exception:
                df = pd.DataFrame()
        if not df.empty and "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def load_metrics_json() -> dict:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                return json.load(f)
        return {}

    def demo_data() -> pd.DataFrame:
        random.seed(int(time.time()) // 10)
        now    = time.time()
        labels = ["normal","normal","normal","DoS","Probe","R2L","U2R"]
        rows   = []
        for i in range(50):
            lbl = random.choice(labels)
            rows.append({
                "id": i+1,
                "timestamp": now - random.randint(0, 3600),
                "src_ip":  f"192.168.1.{random.randint(1,254)}",
                "dst_ip":  f"10.0.0.{random.randint(1,50)}",
                "dst_port": random.choice([22,80,443,3306,8080,21]),
                "attack_type": lbl,
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "severity": ("CRITICAL" if lbl=="U2R" else
                             "HIGH"     if lbl in ("DoS","R2L") else
                             "MEDIUM"   if lbl=="Probe" else "LOW"),
                "acknowledged": 0,
            })
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
                 width=80)
        st.title("🛡️ IDS Control Panel")
        st.markdown("---")
        mode = st.radio("Data source", ["Demo Mode", "Live DB"])
        st.markdown("---")
        st.markdown("### Filters")
        attack_filter = st.multiselect(
            "Attack Types",
            options=["DoS","Probe","R2L","U2R","attack"],
            default=["DoS","Probe","R2L","U2R","attack"],
        )
        severity_filter = st.multiselect(
            "Severity",
            options=["CRITICAL","HIGH","MEDIUM","LOW"],
            default=["CRITICAL","HIGH","MEDIUM","LOW"],
        )
        st.markdown("---")
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=True)
        if st.button("🔄 Refresh Now"):
            st.rerun()
        st.markdown("---")
        st.caption("IBM Internship Project\nML-Based IDS")

    # ── load data ─────────────────────────────────────────────────────────────
    if mode == "Demo Mode":
        df_all = demo_data()
        st.info("📌 Demo Mode — simulated traffic. Switch to 'Live DB' for real data.",
                icon="ℹ️")
    else:
        df_all = load_alerts_db()
        if df_all.empty:
            st.warning("No alerts in DB yet. Start live capture or switch to Demo Mode.")

    df = df_all[
        df_all["attack_type"].isin(attack_filter + ["normal"]) &
        df_all["severity"].isin(severity_filter + ["LOW"])
    ] if not df_all.empty else df_all

    attacks_df = df[df["attack_type"] != "normal"] if not df.empty else df

    # ── header ────────────────────────────────────────────────────────────────
    st.title("🛡️ Network Intrusion Detection System")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("---")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total   = len(df)
    n_att   = len(attacks_df)
    rate    = (n_att / total * 100) if total else 0
    crit    = len(attacks_df[attacks_df["severity"]=="CRITICAL"]) if not attacks_df.empty else 0

    c1.metric("📦 Total Connections", f"{total:,}")
    c2.metric("🚨 Attacks Detected",  f"{n_att:,}",
              delta=f"{rate:.1f}% of traffic", delta_color="inverse")
    c3.metric("🔴 Critical Alerts",   f"{crit:,}")
    c4.metric("✅ Normal Traffic",     f"{total - n_att:,}")
    st.markdown("---")

    # ── charts ────────────────────────────────────────────────────────────────
    cc1, cc2 = st.columns(2)
    _layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font_color="#e0e0e0", margin=dict(t=10, b=10))

    with cc1:
        st.subheader("📊 Attack Type Distribution")
        if not df.empty:
            cnts = df["attack_type"].value_counts().reset_index()
            cnts.columns = ["Attack Type","Count"]
            fig = px.pie(cnts, names="Attack Type", values="Count",
                         color="Attack Type", color_discrete_map=ATTACK_COLORS,
                         hole=0.45)
            fig.update_layout(**_layout, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    with cc2:
        st.subheader("📈 Attack Timeline")
        if not attacks_df.empty and "time" in attacks_df.columns:
            tl = (attacks_df.set_index("time")
                  .resample("10min")["attack_type"].count()
                  .reset_index())
            tl.columns = ["Time","Count"]
            fig2 = px.area(tl, x="Time", y="Count",
                           color_discrete_sequence=["#ff4b4b"])
            fig2.update_layout(**_layout)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No timeline data.")

    # ── severity bar ──────────────────────────────────────────────────────────
    st.subheader("🎯 Severity Breakdown")
    if not attacks_df.empty:
        sc = attacks_df["severity"].value_counts().reindex(
            ["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0).reset_index()
        sc.columns = ["Severity","Count"]
        fig3 = px.bar(sc, x="Severity", y="Count", color="Severity",
                      color_discrete_map={"CRITICAL":"#ff1744","HIGH":"#ff6d00",
                                          "MEDIUM":"#ffd600","LOW":"#69f0ae"},
                      text="Count")
        fig3.update_layout(**_layout, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── alerts table ──────────────────────────────────────────────────────────
    st.subheader("🚨 Recent Alerts")
    if not attacks_df.empty:
        cols = ["id","time","src_ip","dst_ip","dst_port",
                "attack_type","severity","confidence"]
        avail  = [c for c in cols if c in attacks_df.columns]
        recent = attacks_df[avail].head(20)

        def _color(val):
            return {
                "CRITICAL": "background-color: rgba(255,23,68,0.3)",
                "HIGH":     "background-color: rgba(255,109,0,0.3)",
                "MEDIUM":   "background-color: rgba(255,214,0,0.2)",
            }.get(val, "")

        st.dataframe(recent.style.map(_color, subset=["severity"]),
                     use_container_width=True, height=400)
    else:
        st.success("✅ No attacks in selected filters.")

    st.markdown("---")

    # ── model performance ─────────────────────────────────────────────────────
    st.subheader("🤖 Model Performance")
    metrics = load_metrics_json()
    if metrics:
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",            f"{metrics.get('accuracy',0)*100:.2f}%")
        m2.metric("F1 Score (weighted)", f"{metrics.get('f1_weighted',0)*100:.2f}%")
        if "report" in metrics:
            rdf = pd.DataFrame(metrics["report"]).T
            if "support" in rdf.columns:
                rdf["support"] = rdf["support"].astype(int)
            st.dataframe(rdf.style.format("{:.2f}", na_rep="—"),
                         use_container_width=True)

    ic1, ic2 = st.columns(2)
    with ic1:
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, caption="Confusion Matrix", use_container_width=True)
    with ic2:
        if os.path.exists(FI_PATH):
            st.image(FI_PATH, caption="Feature Importance", use_container_width=True)

    if not metrics and not os.path.exists(CM_PATH):
        st.info("Run `python app.py --train` to generate model metrics.")

    # ── top attacking IPs ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌐 Top Attacking Source IPs")
    if not attacks_df.empty and "src_ip" in attacks_df.columns:
        top = (attacks_df.groupby("src_ip").size()
               .reset_index(name="Attack Count")
               .sort_values("Attack Count", ascending=False).head(10))
        fig4 = px.bar(top, x="Attack Count", y="src_ip", orientation="h",
                      color="Attack Count", color_continuous_scale="Reds")
        fig4.update_layout(**_layout, yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    # ── auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(10)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  §7  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def _is_streamlit() -> bool:
    """True when this file is executed by the Streamlit runtime."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    # Launched via:  python -m streamlit run app.py
    _run_dashboard()

else:
    # Launched directly:  python app.py [--preprocess] [--train] [--capture]
    parser = argparse.ArgumentParser(
        description="ML-Based IDS — single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --preprocess
  python app.py --train
  python app.py --train --binary --cross-val
  python app.py --capture --iface "Wi-Fi"
  python -m streamlit run app.py
        """,
    )
    parser.add_argument("--preprocess",    action="store_true", help="Preprocess NSL-KDD data")
    parser.add_argument("--train",         action="store_true", help="Train the Random Forest model")
    parser.add_argument("--capture",       action="store_true", help="Start live packet capture")
    parser.add_argument("--binary",        action="store_true", help="Binary classification mode")
    parser.add_argument("--cross-val",     action="store_true", help="Run 5-fold CV before training")
    parser.add_argument("--n-estimators",  type=int, default=100)
    parser.add_argument("--iface",         default="eth0", help="Network interface for capture")
    parser.add_argument("--count",         type=int, default=0, help="Packets to capture (0=infinite)")
    args = parser.parse_args()

    if args.preprocess:
        preprocess(binary=args.binary)

    elif args.train:
        run_training(binary=args.binary,
                     n_estimators=args.n_estimators,
                     cross_val=args.cross_val)

    elif args.capture:
        if not SCAPY_AVAILABLE:
            print("Scapy not installed.  pip install scapy")
            sys.exit(1)
        predictor     = IDSPredictor(binary=args.binary)
        alert_manager = AlertManager()

        def _on_result(result, features):
            icon = "🚨" if result.is_attack else "✅"
            print(f"{icon} [{result.label}] "
                  f"{features['_src_ip']} → {features['_dst_ip']}:{features['_dst_port']} "
                  f"conf={result.confidence:.2f}")

        cap = LiveCapture(args.iface, _on_result, alert_manager, predictor)
        try:
            cap.start(count=args.count)
        except KeyboardInterrupt:
            cap.stop()
            print("\nCapture stopped.")

    else:
        parser.print_help()
