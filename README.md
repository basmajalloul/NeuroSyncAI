# NeuroSyncAI – AI-Powered Cognitive & Motor Function Analysis 🧠  

## 📌 Overview  
**NeuroSyncAI** is an AI-driven **multimodal cognitive and motor function assessment tool** that integrates **EEG, HRV, and pose data**. It provides **real-time analysis, anomaly detection, and automated insights** to support research in neurocognitive and motor function studies.

### **🖥 NeuroSyncAI Dashboard (dashboard_yolo_synthetic.py)**  
An interactive **Dash-based web application** for visualizing EEG, HRV, and pose data, supporting **real-time annotations, anomaly detection, and LLM-powered insights**.

#### 🔹 **Key Features**  
✅ **Real-Time EEG, HRV & Pose Synchronization** – View multimodal physiological data in a structured interface.  
✅ **AI-Powered Annotations** – Detect anomalies like cognitive stress, fatigue, and movement irregularities.  
✅ **Automated Insights using LLMs** – Ask research-based queries about EEG, HRV, and pose correlations.  
✅ **Interactive Visualization** – Use range sliders and overlays to analyze data dynamically.  
✅ **Ontology-Based Analysis** – Leverages structured knowledge to enhance AI-driven insights.

#### ⚙ **Tech Stack**  
- **Python, Dash, Flask** – Backend and interactive dashboard  
- **Plotly, Matplotlib** – Data visualization  
- **YOLO, Mediapipe** – Pose estimation and motion tracking  
- **Scikit-learn, TensorFlow** – Machine learning models  
- **OpenAI GPT-based LLMs** – Natural language insights  
- **RDFLib, JSON-LD** – Ontology-based data structuring  

---

### **📊 NeuroSyncAI LLM Validation (NeuroSyncAI_LLM_validation.ipynb)**  
A **Jupyter Notebook** for validating the **LLM-powered insights** in NeuroSyncAI. It evaluates how well **large language models (LLMs)** interpret EEG, HRV, and pose data to provide meaningful scientific explanations.

#### 🔹 **Key Features**  
✅ **Validates AI-Generated Annotations & Insights**  
✅ **Evaluates LLM’s Accuracy in Cognitive & Motor Function Analysis**  
✅ **Compares AI vs. Expert Analysis on Neurophysiological Data**  

#### ⚙ **Tech Stack**  
- **Python, Jupyter Notebook**  
- **OpenAI GPT-4 API** for AI-powered insights  
- **Pandas, NumPy, Scikit-learn** for data handling  
- **Matplotlib, Seaborn** for visualization  

---

## 📥 **Installation & Usage**  
### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```
*(Ensure you have Python 3.8+ installed.)*

### **2️⃣ Run the NeuroSyncAI Dashboard**  
```bash
python dashboard_yolo_synthetic.py
```
*(Access the web interface via `http://127.0.0.1:8052/`.)*

### **3️⃣ Run LLM Validation Notebook**  
Open **NeuroSyncAI_LLM_validation.ipynb** in Jupyter Notebook and execute the cells.

---

## 📄 **License**  
This project is licensed under the **MIT License**.

---

## 📧 **Contact & Contributions**  
For inquiries, contributions, or collaboration, reach out via **GitHub Issues** or email.

---
