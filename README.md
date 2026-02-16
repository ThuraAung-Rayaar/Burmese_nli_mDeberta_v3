# mmNLI: Burmese Text Natural Language Inference with microsoft-DeBERTa-V3

A **Burmese NLI model** fine-tuned from mdeberta-v3-base, trained on a carefully cleaned and curated dataset, achieving **76%** accuracy on the test set.

This model predicts the relationship between a **premise** and a **hypothesis** as one of:

* **Entailment**
* **Neutral**
* **Contradiction**
## Model & Demo
- Model >> https://huggingface.co/Raayar/Burmese_nli_mDeBERTa_V3
- Space Demo Link >> https://huggingface.co/spaces/Raayar/Burmese_nli_mdeberta_v3

---
## Model Details

* **Base model:** `mdeberta-v3-base`
* **Language:** Burmese (Myanmar)
* **Task:** Natural Language Inference (NLI)
* **Labels:** `entailment`, `neutral`, `contradiction`
* **Framework:** Transformers / PyTorch
---
## Dataset and its Structure

The dataset consists of **~10k [10,443] samples** across three classes:

| Label | Class         | Count |
| ----: | ------------- | ----: |
|     0 | Entailment    | 3,608 |
|     1 | Neutral       | 3,466 |
|     2 | Contradiction | 3,369 |

and the dataset is prepared from:

* By clearning Burmese NLI data (source: *[(https://huggingface.co/datasets/akhtet/myanmar-xnli)]*) and additional **manually created** samples.
* **Translated English NLI**  data for diversity.
* Most samples follow a **1 premise → 3 hypotheses** structure. Each hypothesis has a **different NLI label**.
* An  **`genre`** field is included intended for **future zero-shot / cross-genre experiments** later.
## Preprocessing

Steps:
1. **Whitespace Removal:** All spaces are stripped from the input
2. **Punctuation Cleaning:** Burmese sentence-ending markers (။) are removed
3. **Unicode normalization** (NFC)
4. **Zawgyi detection**
5. **Automatic conversion to Unicode** if Zawgyi text is detected
6. Rely on **XLM-R subword tokenizer** for tokenization

## Data Splitting Strategy

To prevent data leakage caused by shared premises:

* **Train:** 70%
* **Validation:** 15%
* **Test:** 15%

Instead of random shuffling:

* **GroupShuffleSplit** is used with `n_splits=1` and `random_state=127`
* Samples are grouped by the **'premise'** and Samples  with the **same premise always stay in the same split**
* Prevents:

  * Premise overlap across splits
  * Hypothesis leakage between train / validation / test sets
---


## Training Setup

The model was trained using **Hugging Face Transformers** with the following key configurations:

* **Model:** microsoft/mdeberta-v3-base
* **Target Labels:** 3 classes (`entailment`, `neutral`, `contradiction`)
* **Epochs:** Up to **15 epochs**
* **Early Stopping:** Enabled (patience = 1)
* **Learning Rate:** `1e-5`
* **Batch Size:** `16` (train & evaluation)
* **Weight Decay:** `0.01`
* **Warmup Ratio:** `0.2`
* **FP16 Training:** Enabled
* **Best Model Selection:** Based on **F1-score**
* **Seed:** `127`

The best checkpoint was automatically loaded at the end of training using **early stopping** and **F1-based model selection**.

---

## Evaluation Metrics

he model performance is evaluated using a comprehensive suite of metrics to account for different dataset distributions:

* **Accuracy:** Used as the primary metric for the balanced dataset
* **Macro F1-score:** Used to evaluate performance if the dataset is skewed
* **Precision:** Monitored for cases where False Positives are fatal
* **Recall:** Monitored for cases where False Negatives are fatal
* **Error Rate:** Calculated as $1 - \text{Accuracy}$ to pinpoint failure frequency

## Training Results

|    Epoch | Train Loss |   Val Loss |   Accuracy |         F1 |  Precision |     Recall |
| -------: | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
|        1 |     1.0644 |     1.0814 |     0.3986 |     0.3331 |     0.3960 |     0.3977 |
|        2 |     0.9610 |     0.9564 |     0.4861 |     0.4117 |     0.4557 |     0.4907 |
|        3 |     0.7905 |     0.8457 |     0.5735 |     0.5349 |     0.5752 |     0.5750 |
|        4 |     0.7204 |     0.7521 |     0.6724 |     0.6693 |     0.6700 |     0.6752 |
|        5 |     0.5888 |     0.6582 |     0.7161 |     0.7150 |     0.7156 |     0.7173 |
|        6 |     0.5746 |     0.6133 |     0.7503 |     0.7509 |     0.7534 |     0.7507 |
|        7 |     0.5937 |     0.5978 |     0.7579 |     0.7570 |     0.7581 |     0.7589 |
|        8 |     0.3793 |     0.6480 |     0.7687 |     0.7691 |     0.7692 |     0.7694 |
|        9 |     0.4497 |     0.6606 |     0.7598 |     0.7582 |     0.7664 |     0.7601 |

Training stopped early at **epoch 9** due to Accuracy rate plateau and the best model saved is at the f1 score of 0.7691 at ecpoch 8.


## Test Set Performance

```json
{
  "test_loss": 0.7139414548873901,
  "test_accuracy": 0.756774193548387,
  "test_f1": 0.7557812004382494,
  "test_precision": 0.7578269693244604,
  "test_recall": 0.7563696801214403,
  "test_runtime": 3.5405,
  "test_samples_per_second": 437.786,
  "test_steps_per_second": 27.397
}
```

## Confusion Matrix on Test Set

Label order: **entailment, neutral, contradiction**

```
[[426  55  54]
 [ 115 342  54]
 [ 42  57 405]]
```

Rows represent **true labels**, columns represent **predicted labels**.

## Classification Report (Test Set)

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Entailment       | 0.73      | 0.80   | 0.76     | 535     |
| Neutral          | 0.75      | 0.67   | 0.71     | 511     |
| Contradiction    | 0.79      | 0.80   | 0.80     | 504     |
| **Accuracy** |           |        | **0.76** | 1550    |
| **Macro Avg** | 0.76      | 0.76   | 0.76     | 1550    |
| **Weighted Avg** | 0.76      | 0.76   | 0.76     | 1550    |
----

## Inference Example

You can use the model as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import unicodedata

model_name = "Raayar/Burmese_nli_DeBERTa_V3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def convert(text):
    text = text.replace(" ", "")
    text = text.replace("။", "")
    return unicodedata.normalize('NFC', text)

premise = "လက်ရှိ ကမ္ဘာလုံးဆိုင်ရာ ရာသီဥတု ပြောင်းလဲမှု ဖြစ်စဉ်တွေကြောင့် ပင်လယ်ရေမျက်နှာပြင် မြင့်တက်လာပြီး ကမ်းရိုးတန်း ဒေသတွေမှာ နေထိုင်တဲ့ လူဦးရေ   သန်းပေါင်းများစွာဟာ ရေဘေးအန္တရာယ်နဲ့ အိုးအိမ်စွန့်ခွာရမယ့် အခြေအနေတွေကို ရင်ဆိုင်နေကြရပါတယ်။"
hypothesis = "သဘာဝပတ်ဝန်းကျင် ဖောက်ပြန်ပျက်စီးလာတာဟာ ကမ်းခြေအနီးမှာ နေထိုင်သူတွေအတွက် အသက်အန္တရာယ်နဲ့ နေထိုင်မှု ဘဝတွေကို ခြိမ်းခြောက်နေပါတယ်။"

inputs = tokenizer(
    convert(premise),
    convert(hypothesis),
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
).to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)

predicted_class = torch.argmax(outputs.logits, dim=1).item()

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
print("Predicted label:", label_map[predicted_class])
# conf
probs = torch.softmax(outputs.logits, dim=-1)[0]
print("Confidence:", {k: round(float(probs[i]), 3) for i, k in label_map.items()})


```
---
## Our Team

This project is part of our Semester-8 **CS-502 Natural Language Processing** course.

| Name            | Role   | GitHub |
|-----------------|--------|--------|
| Yoon Thiri Aung | Leader | https://github.com/yoon-thiri04 |
| Soe Sett Lynn   | Member | https://github.com/ssettlynn |
| Thura Aung      | Member | https://github.com/ThuraAung-Rayaar |

## Limitations & Future Work

* Genre-aware and **zero-shot classification** is planned but not yet implemented
* Performance may vary for:

  * Very long inputs
  * Out-of-domain or highly informal Burmese
* Future improvements:

  * Larger native Burmese NLI dataset
  * Explicit genre-based evaluation
  * Domain adaptation

---
