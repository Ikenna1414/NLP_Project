# LLM Project: IMDb Sentiment Classifier with DistilBERT

---

## Project Task

This project fine-tunes a pre-trained language model to **predict the sentiment (positive or negative)** of movie reviews from the IMDb dataset. The goal is to leverage transformer models for accurate text classification using **Hugging Face Transformers**.

---

## Dataset

- **Name**: IMDb Movie Reviews
- **Source**: Hugging Face Datasets (`imdb`)
- **Description**: 50,000 movie reviews (25k train, 25k test), equally split between positive and negative sentiments.
- **Properties**:
  - Binary classification: `0 = Negative`, `1 = Positive`
  - Reviews are varied in length, requiring truncation for transformer compatibility

---

## Pre-trained Model

- **Model**: `distilbert-base-uncased`
- **Why Chosen**:
  - Lightweight, fast version of BERT
  - Pre-trained on large English corpus
  - Great trade-off between accuracy and speed
- **Modifications**:
  - Fine-tuned on IMDb data
  - Truncated inputs to 512 tokens
  - Saved and deployed using Hugging Face Inference API

---

## Performance Metrics

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 92.5%   |
| F1 Score      | 92.3%   |
| Precision     | 92.1%   |
| Recall        | 92.4%   |
| AUC-ROC       | 95.1%   |

Baseline (TF-IDF + Logistic Regression): ~86% accuracy

---

## Hyperparameters

Key hyperparameters used in optimization:

- `learning_rate`: `2e-5` (also tested `3e-5`, `5e-5`)
- `num_train_epochs`: `3`
- `per_device_train_batch_size`: `16`
- `per_device_eval_batch_size`: `32`
- `weight_decay`: `0.01`
- `logging_steps`: `100`
- `evaluation_strategy`: `epoch`
- `push_to_hub`: `True`

Used `Trainer` API from Hugging Face for fine-tuning with `load_best_model_at_end=True`.

---

## Training Workflow Overview

```text
IMDb Dataset → Preprocessing & Truncation → Tokenization → Fine-tune DistilBERT →
Evaluate & Optimize → Push to Hugging Face Hub → Deploy via Inference API
```


## Ethical Considerations

As part of the project, we’ve applied ethical thinking throughout development:

###  1. Data Anonymity & Safety
- Used the **IMDb dataset**, which contains publicly available, anonymized movie reviews.
- Avoided user-identifiable or sensitive data.

###  2. Bias Awareness
- Acknowledged that **movie sentiment** can reflect subjective reviewer bias.
- Selected **DistilBERT** (a general-purpose LLM) and monitored predictions for **toxic behavior** or **bias propagation**.

###  3. Transparent Reporting
- Shared **model limitations** and intended use in the Hugging Face model card.
- Made the model **publicly accessible** (with **read-only API access**) for education and safe experimentation.


##  Folder Structure
```
.
├── data/
│   └── imdb_cleaned.csv
├── notebooks/
│   ├── 1-data-load.ipynb
│   ├── 2-representation.ipynb
│   ├── 3-transformers.ipynb
│   ├── 4-optimization.ipynb
│   └── 5-deployment.ipynb
├── model/
│   └── distilbert-imdb-optimized/
├── README.md
```


## Future Work

- **Add Explainability**  
  Integrate tools like SHAP or LIME to interpret model predictions and explain how different tokens contribute to the sentiment classification.

- **Support for Multilingual Sentiment**  
  Extend the model to handle non-English movie reviews by fine-tuning multilingual transformers like `xlm-roberta-base`.

- **Adversarial Testing**  
  Evaluate model robustness against adversarial or ambiguous inputs to test stability under edge cases.

- **Gradio Web App Interface**  
  Build a simple and interactive Gradio interface for real-time model predictions and deployment showcase.

- **Continual Fine-Tuning**  
  Periodically fine-tune the model with newer IMDb reviews or user-generated content to adapt to language and sentiment drift.

- **Model Distillation for Speed**  
  Explore model distillation techniques to reduce inference time and deploy on low-resource environments or mobile devices.

---

## Summary

This project demonstrates the complete pipeline of fine-tuning and deploying a transformer-based sentiment classifier using the IMDb movie review dataset. Starting from text preprocessing and representation (BoW/TF-IDF) through transformer tokenization, we fine-tuned `distilbert-base-uncased` using Hugging Face’s `Trainer` API and deployed the model via the Hugging Face Inference API.

Throughout the process, we emphasized:
- Ethical data usage
- Robust evaluation and performance tracking
- Scalable and accessible deployment

The project showcases how modern NLP tools can be integrated end-to-end with minimal infrastructure overhead, while still adhering to ethical and reproducible ML standards.
