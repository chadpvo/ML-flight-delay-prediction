# ML-flight-delay-prediction
How we engineered a distributed ML pipeline to predict flight delays at scale—from 1.7M to 30M rows using weather data, operational features, and graph-based network analysis.This is our Machine Learning at Scale's Final Project with Amy Steward, Kristen Lin, Priscilla Siow, and Uma Rao Krishnan.

## Project Overview
This project focuses on predicting airline delays using historical flight data. By analyzing factors such as weather conditions, carrier performance, and temporal patterns, the model aims to classify flights as "Delayed" or "On Time."

The goal is to assist travelers and stakeholders in anticipating disruptions and understanding the primary drivers of flight delays.

**Tech Stack:** Apache Spark 3.x • Databricks • PySpark MLlib • XGBoost • PyTorch (FT-Transformer) • NetworkX • NVIDIA T4 GPU

**Key Results:**
* Winner: **The Feature Token Transformer**.

Huge Credit to Kristen Lin and Amy Steward on leading the heavy data engineering efforts that helped us scale our models

---

## Repository Structure

```text
flight-delay-prediction/
├── EDA/    
├── data-pre-processing-scripts/
├── ML-models-notebook/
└── README.md           
