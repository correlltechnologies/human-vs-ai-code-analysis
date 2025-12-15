# human-vs-ai-code-analysis
This repository contains an empirical analysis of stylistic and structural differences between human-written code and AI-generated code, based on the OSS-forge/HumanVsAICode dataset.  The project is designed to be fully runnable in Google Colab and produce publishable results (plots, tables, insights).

## Dataset

HumanVsAICode
Source: Hugging Face – OSS-forge/HumanVsAICode

Each sample contains:

A code snippet

A label indicating whether it was written by a human or an AI

Notebook 01 — Style & Complexity Analysis

Goal: Quantitatively compare how humans and AI write code.

Metrics Computed

Lines of code (LOC)

Average line length

Comment density

Indentation depth

Function count

Average variable name length

Cyclomatic complexity (via radon)

Outputs

Boxplots comparing human vs AI code

Summary statistics table

Example snippets

How to Run (Google Colab)

Open Google Colab

Upload 01_style_analysis.ipynb

Run all cells

Planned Extensions

Human vs AI classifier (TF-IDF + Logistic Regression)

Code embedding visualization (CodeBERT + UMAP)

Runtime performance benchmarking

License

MIT
