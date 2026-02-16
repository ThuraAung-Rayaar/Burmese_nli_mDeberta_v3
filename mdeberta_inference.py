from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import unicodedata


model_name = "Raayar/Burmese_nli_DeBERTaV3"

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


probs = torch.softmax(outputs.logits, dim=-1)[0]
print("Confidence:", {k: round(float(probs[i]), 3) for i, k in label_map.items()})
