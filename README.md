*A state of the art semi-supervised approach for binary classification tasks.*

**Author: Ahmad Rammal**

# About
This repository introduces a semi-supervised approach applied on binary classification task for topics.
The main advantage of using this approach is the minor amount of labeled data that is required in order to train the models. We’re therefore solving the main constraint behind the supervised approach in NLP: getting better results for less data.

The classical way to deal with binary classification is the supervised approach. By analyzing the training data, a supervised learning algorithm produces an inferred function that can be used for mapping new examples. 

Nonetheless, the constraint with such approach is the large amount of labeled data that is required in order to train a supervised model; especially in the NLP field as it is not easy for computers to understand the rules that govern the passing of information using natural languages.  A human agent is often required in order to create labeled data for a learning problem. Therefore, the cost of labeling is likely to make the acquisition of large, fully labeled training sets unfeasible. 

A semi-supervised learning approach could prove to be very useful in such situations. In addition, the explicit dissociation in those semi-supervised models between the creation of embeddings and the classification task opens the door to an opportunity of using the embedded data on other NLP tasks efficiently.

# Models Overview:

## I. Mean Model

**Overview:**

Using the cosine similarity we’re going to classify a text as 1 (fits the topic) or 0 (does not fit)

We start by finding the text’s vector representation known as embedding using a sentence embedder like SBert. We’ll call this vector representation E\_text.

Now suppose that the topic could also be represented by a vector V. The cosine similarity between E\_text and V would tell us how semantically close the text and the topic are. 

What’s left to do is to define a threshold t above which this cosine similarity implicates that the text fits the topic or not. In other term we take a float t such that:

If cos(E\_text, V) > t then the text is considered “close enough” to the topic. Otherwise the text doesn’t fit the topic.

![Image](https://github.com/rammalahmad/Gram-Schmidt-Model/blob/main/Graphics/Aspose.Words.88ad23da-b2a6-4d11-b15e-e1f4a6e03cd7.001.png)

`          `*Fig. 1: Overview on the Mean Model* 

With that being said, we still need to find the two following parameters:

- The array that represents the topic V
- The threshold t

**Training:**

The training is where our model learns to identify a certain topic. The procedure is divided into two tasks:

- Finding V the vector representing the topic 
- Finding t the threshold above which we classify the text as in the topic

**Finding the topic’s vector representation:**

In order to represent a topic by a vector we start by finding keywords that are most relevant to the topic. Those keywords are the topic’s dictionary. That’s the hardest part of the training since we have to do it by hand. 

Once we have the dictionary we generate the embeddings of that dictionary, then we calculate the average of those embeddings. The outcome will be V the vector representing the topic

`        `![Image](https://github.com/rammalahmad/Gram-Schmidt-Model/blob/main/Graphics/Aspose.Words.88ad23da-b2a6-4d11-b15e-e1f4a6e03cd7.002.png)

*Fig. 2: Finding the topic’s vector representation*

**Finding the threshold:**

As for the threshold, we simply test our model on a training set (a sample of labeled data) using different thresholds ranging from 0 to 1. We save the threshold that provides us with the best accuracy.

![Image](https://github.com/rammalahmad/Gram-Schmidt-Model/blob/main/Graphics/Aspose.Words.88ad23da-b2a6-4d11-b15e-e1f4a6e03cd7.003.png)

*Fig. 3: Finding Mean model’s threshold*

## II. Gram-Schmidt Model

**Overview:**

In order to further improve our previous model we could consider generalizing the mathematical tools that were applied. 

In our case instead of representing a topic by a simple vector we’re going to represent it by a vector space. More precisely in the implementation we will use a set of vectors which constitute the orthonormal basis of the vector space

Now for a certain text, just like before we start by finding the text’s vector representation E\_text.

Since we have a vector space representing the topic now instead of a simple vector, in order to evaluate how close E\_text is to that plane we start by finding the orthonormal projection p of E\_text on our vector space. Then we calculate the cosine between E\_text and p. This value would tell us how close is E\_text is to the topic’s vector space, thus in a way we find out how semantically close is the text to the topic.

![Image](https://github.com/rammalahmad/Gram-Schmidt-Model/blob/main/Graphics/Aspose.Words.88ad23da-b2a6-4d11-b15e-e1f4a6e03cd7.004.png)

*Fig. 4: The orthogonal projection of E\_text on the Plane representation of the topic*

Finally, just like the previous model we’ll need a threshold t above which this cosine similarity implicates that the text fits the topic or not. 

![Image](https://github.com/rammalahmad/Gram-Schmidt-Model/blob/main/Graphics/Aspose.Words.88ad23da-b2a6-4d11-b15e-e1f4a6e03cd7.005.png)

`          `*Fig. 5: Overview on the Gram-Schmidt Model* 

**Training:**

The procedure is also divided into two tasks:

- Finding S, an orthonormal basis for a space vector representation of the topic 
- Finding t the threshold above which we classify the text as in the topic

**Finding an orthonormal basis for a vector space representation of the topic**

We start by looking for keywords that are relevant to the topic. These keywords constitute the topic’s dictionary. We then calculate the embeddings of those keywords. Unlike the mean model we won’t just find the average of those embeddings. Instead we will consider that those vectors generate the vector space representation of the topic. 

Therefore the vector space representation of our topic is nothing other than the vector space generated by the embeddings of the topic’s dictionary.

We need however an orthonormal basis of our vector space in order to calculate the orthonormal projection later on… That’s why we use the Gram-Schmidt process in order to create an orthonormal basis from the regular one we found earlier. The outcome is indeed what we were looking for: a set of vectors that form an orthonormal basis for a vector space representation of the topic.

`                     `**![Image](https://github.com/rammalahmad/Gram-Schmidt-Model/blob/main/Graphics/Aspose.Words.88ad23da-b2a6-4d11-b15e-e1f4a6e03cd7.006.png)**

*Fig. 6: Finding an orthonormal base for the plane representation of the topic*

**Finding the threshold:**

Exactly like the previous model, we test our model on a training set using different thresholds ranging from 0 to 1. We save the threshold that provides us with the best accuracy.

![Image](https://github.com/rammalahmad/Gram-Schmidt-Model/blob/main/Graphics/Aspose.Words.88ad23da-b2a6-4d11-b15e-e1f4a6e03cd7.007.png)

*Fig. 7: Finding Gram-Schmidt model’s threshold*

## Advantages of this structure
The main advantage of such structure is that it only relies on a generic Deep Learning model like SBERT therefore exceeding the need for a classical finetuning process. SBERT is thus sufficient for the creation of a binary classifier independently from the topic at hand.

In addition, this model dissociates the creation of the embeddings from the classification task. The generated embeddings could be saved and used again on another binary classifier task or even employed on other NLP tasks such as clustering. This would greatly save time since the creation of the embeddings is the most costly part of the process.

Now that we’ve established the multiple benefits of using this model we need to evaluate its performance on a classication task.


On the example of the environment we get the following metrics

||FinBERT|MEAN|GS|
| - | - | - | - |
|Accuracy|0.903|0.849|**0.926**|
|Precision|0.924|0.832|**0.939**|
|Recall|0.877|0.873|**0.910**|
|F score|0.900|0.852|**0.924**|

                          

Overall the metrics above suggest that the GS model clearly outperforms FinBERT in this task. The Mean model scores below the latter but still gives fairly good results. Thus the semi-supervised approach proves itself to be an adequate alternative to the classical supervised approach.
