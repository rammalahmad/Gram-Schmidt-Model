from typing import List
from transformers import pipeline, BertTokenizer, BertForSequenceClassification

class Finbert_model:
    def __init__(self):
        """
        # Info
        ---
        This is the FinBERT Model, it's an supervised binary classifier for text
        """
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
        self.filter = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

    def fit(self, documents:List[str])->List[int]:
        """
        # Info
        ---
        Classifies text into 1 or 0 (1=relevant to topic, 0=not relevant)

        # Params
        ---
        documents: List of string containing the documents we want to classify

        # Returns
        ---
        boolean list (1=relevant to topic, 0=not relevant)
        """ 
        l = self.filter(documents)
        result = []
        for t in l:
            if t['label']=="Environmental":
                result.append(1)
            else:
                result.append(0)
        return result