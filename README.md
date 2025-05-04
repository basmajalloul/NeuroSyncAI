
# NeuroSyncAI – AI-Powered Cognitive & Motor Function Analysis 🧠

## 📌 Overview
**NeuroSyncAI** is an AI-driven **multimodal cognitive and motor function assessment tool** that integrates **EEG, HRV, and pose data**. It provides **real-time analysis, anomaly detection, and automated insights** to support research in neurocognitive and motor function studies.

### 🖥 NeuroSyncAI Dashboard (`dashboard_yolo_synthetic.py`)
An interactive **Dash-based web application** for visualizing EEG, HRV, and pose data, supporting **real-time annotations, anomaly detection, and LLM-powered insights**.

#### 🔹 Key Features
✅ **Real-Time EEG, HRV & Pose Synchronization**  
✅ **AI-Powered Annotations**  
✅ **Automated Insights using LLMs**  
✅ **Interactive Visualization**  
✅ **Ontology-Based Analysis**

#### ⚙ Tech Stack
- Python, Dash, Flask  
- Plotly, Matplotlib  
- Scikit-learn, TensorFlow  
- OpenAI GPT-based LLMs  
- RDFLib, JSON-LD

---

## 📊 LLM-Based Validation Studies

We validated NeuroSyncAI’s LLM-based reasoning pipeline using three independent datasets:

### ✅ `LLM_EEG_HRV_Pose_validation_neuropose.ipynb`
Validation using the **NeuroPose** synthetic multimodal dataset (EEG, HRV, Pose).  
📁 LLM results: `llm_subject_predictions_latest_neuropose.csv`  
🔓 Dataset: [NeuroPose (OSF)](https://osf.io/sc5v2/)

### ✅ `LLM_EEG_HRV_validation_boudaya.ipynb`
Validation using **real-world EEG+HRV** data collected under cognitive stress/fatigue protocols.  
📁 LLM results: `llm_subject_predictions_eeg_hrv.csv`  
📬 Dataset available **upon request** (not publicly distributable).

### ✅ `LLM_Pose_validation_stroke.ipynb`
Validation using **public pose data** from a gait analysis study in stroke patients.  
📁 LLM results: `llm_subject_predictions_stroke.csv`  
🔓 Dataset: [Springer Nature Gait Dataset](https://doi.org/10.6084/m9.figshare.c.6503791.v1)

Each validation notebook runs the LLM inference pipeline, parses predictions, and computes accuracy, recall, precision, and confidence distributions to assess decision consistency.

---

## 📥 Installation & Usage
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Dashboard
```bash
python dashboard_yolo_synthetic.py
```

### 3️⃣ Run LLM Validation Notebooks
Open and run the notebooks in your preferred environment (Jupyter Lab or Notebook), you need an OpenAI API key to execute:
- `LLM_EEG_HRV_Pose_validation_neuropose.ipynb`
- `LLM_EEG_HRV_validation_boudaya.ipynb`
- `LLM_Pose_validation_stroke.ipynb`

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 📧 Contact & Contributions
For inquiries, collaboration, or to request datasets, please reach out via **GitHub Issues** or email.

---
