# 🌿 GreenAI Email Classifier

> ⚡ A sustainable, energy-efficient AI system that classifies emails while tracking energy usage and CO₂ emissions.  
> Combines **TF-IDF**, **DistilBERT**, and **DeBERTa-v3** models in a smart *cascade* architecture to balance accuracy and environmental impact. 🌍

---

## 🧠 Overview

The **GreenAI Email Classifier** is designed to classify emails into categories like:
- 📧 `work`
- 🚫 `spam`
- 💬 `support`

It uses a **cascade of three models** — Green (light), Medium, and Heavy — that are selected dynamically based on confidence thresholds, minimizing computational cost.

### 🌱 Key Features
- ⚙️ **Cascade Classifier** with three tiers of models (TF-IDF → DistilBERT → DeBERTa-v3)
- 💡 **Automatic model switching** based on confidence
- 🔋 **Energy and CO₂ tracking**
- 🧾 **SQLite database** for inference logs and metrics
- 🚀 **FastAPI** backend for real-time inference
- 📊 **Streamlit dashboard** for monitoring and visualization
- 🐳 **Dockerized** setup for full-stack deployment

---

## 🧩 Architecture

```mermaid
graph TD
A[Client / Dashboard] -->|REST API| B[FastAPI Backend]
B -->|Predict| C[Green Model 🌱]
B -->|Escalate| D[Medium Model ⚙️]
B -->|Escalate| E[Heavy Model 🚀]
B -->|Log| F[(SQLite Database)]
B -->|Track| G[Energy Tracker ⚡]
