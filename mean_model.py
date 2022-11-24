from xmlrpc.client import Boolean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np
import pandas as pd
import operator

class Mean_model:
    def __init__(self, topic_dict:List[str], train_df: pd.DataFrame, embed_test:Boolean = True):
        """
        # Info
        ---
        This is the Mean Model, it's an unsupervised binary classifier for text
        # Params
        ---
        topic_dict: list of string containing the collected dictionnary on the topic at hand
        train_df: pandas dataframe the supervised colleceted Dataset
        embed_test:Boolean Whether or not the text is already embedded into vectors
        """

        self.topic_dict = topic_dict
        self.train_df = train_df
        self.embed_test = embed_test


    def test(self, test_df: pd.DataFrame = None, test_embed:np.array = None):
        """
        # Info
        ---
        Function to test the model on test dataset
        # Params
        ---
        test_df: pandas dataframe, the database we wanna do the test on
        test_embed : Numpy array containing the embedded vectors

        # Return
        ---
        Binary vector (1 == is in Topic, 0 == not in Topic)
        """

        if self.embed_test:
            test_text = test_df['text'].to_list()
            test_embeddings = self.embed(test_text)
        else:
            test_embeddings = test_embed 
        test_cos_sim = [cosine_similarity(a.reshape(1,-1), self.topic_vector.reshape(1,-1)) for a in test_embeddings]
        return [1 if e>self.threshold else 0 for e in test_cos_sim]

    def train(self):
        """
        # Info
        ---
        Function to find the model's parameters using the train dataset and the topic's dictionnary
        """
        self.topic_vector = self.find_topic_vector()
        self.threshold = self.find_threshold()

    def find_topic_vector(self)->np.array:
        """
        # Info
        ---
        Finds the mean model's topic vector using the topic's dictionnary
        """
        embedded_dict = self.embed(self.topic_dict)
        return np.mean(embedded_dict, axis=0)

    def find_threshold(self):
        """
        # Info
        ---
        Finds the mean model's threshold using the train dataset 
        """
        train_label = self.train_df['label'].to_list()
        train_text = self.train_df['text'].to_list()
        train_embeddings = self.embed(train_text)
        train_cos_sim = [cosine_similarity(a.reshape(1,-1), self.topic_vector.reshape(1,-1)) for a in train_embeddings]
        thresholds = [i/100 for i in range(1,95)]
        accuracy_dict = {}
        for threshold in thresholds:
            prediction = [1 if e>threshold else 0 for e in train_cos_sim]
            acc_list = [prediction[i] == train_label[i] for i in range(len(prediction))]
            accuracy_dict[threshold] = sum(acc_list)/len(prediction)
        return max(accuracy_dict.items(), key=operator.itemgetter(1))[0]

    @staticmethod
    def embed(documents:List[str])->np.ndarray:
        """
        # Info
        ---
        Embed text into array 
        # Params
        ---
        documents: list of string text

        # Return
        ---
        numpy array of embedded text
        """
        model = SentenceTransformer('all-mpnet-base-v2')
        return np.array(model.encode(documents, batch_size=32, show_progress_bar=True))


def environment_train():
    from topic_dict import topic_dict
    train_df = pd.read_csv('train_data.csv')
    model = Mean_model(topic_name='environment', topic_dict=topic_dict, train_df=train_df)
    model.train()