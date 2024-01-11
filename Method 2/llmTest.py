# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
# model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

from transformers import pipeline

pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
res = pipe("I am happy")
print(res)

