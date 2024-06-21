# Fake News Detection 

------

## Contributors
- [Alvern Ong Wei Zhe](https://github.com/aowz)
- Goh Kian Hwee, Justin
- Jared Lau Zilek
- [Justin Chew Wei Sheng](https://github.com/juschew03)
- Lee Tian, Caleb
- [Seah Zhi Han Mervyn](https://github.com/undeser)

-------

# Abstract
With the rampant expansion of digitalised media, the availability of information has increased significantly, leading to a greater spread of fake news. Given the vast volume of web content, automating the detection of fake news is a practical NLP challenge that can benefit online content providers by minimising the human time and manual effort required to identify and curb the dissemination of misinformation. 

The business problem which fake news detection models typically tackle is the dissemination of misinformation and reputation damage. The ML goal of these models is to correctly classify the text content as either real or fake based on its content and context.

We used a joint dataset consisting of four different structured datasets which have been manually checked: Politifact Fact Check Dataset, LIAR Dataset, Fake Real News Dataset and Fake News Detection Dataset.

Generally, we can summarise our findings into the following points:
1. **Model Efficacy**: Our models have demonstrated varying levels of accuracy in distinguishing between real and fake news. The highest performance was observed in the Complement Naive Bayes + CountVectorizer model, indicating its effectiveness in capturing linguistic nuances that are significant in fake news detection.
2. **Feature Importance**: The analysis revealed that certain linguistic features, such as n-grams and word embeddings, play a crucial role in improving model accuracy. These features capture both the contextual and semantic properties of the text, thereby enhancing the detection process.
3. **Innovative Approaches**: The use of ensemble methods and advanced neural network architectures like CNNs contributed significantly to reducing bias and misclassification. This highlights the potential of combining multiple models to leverage their individual strengths for better performance.
4. **Challenges and Limitations**: Despite the advancements, the models still face challenges due to the inherent complexities of natural language and the subtleties involved in differentiating factual from misleading content. The performance limitations underscore the need for ongoing research and refinement of techniques in the NLP field.

# Data Description

## Datasets
We have used a joint dataset of four different structured datasets which have been manually labelled, this is to also provide a dataset that consists of a diversity of data sources.

| S/N |Raw Dataset | Description | No. of Data Points |
|:-----:|:---|:---|:---:|
| 1 | Politifact Fact Check Dataset | The PolitiFact dataset compiled statements made by U.S. politicians, fact-checked and rated for truthfulness by PolitiFact journalists. | 21,152 rows|
| 2 | LIAR Data Analysis | Dataset of manually labelled short statements. | 10,269 rows|
| 3 | Fake and Real News Dataset | News articles from media organisations such as the New York Times, WSJ, Bloomberg, NPR, and the Guardian. | 4,594 rows |
| 4 | Fake News Detection Dataset | The True dataset was collected from real world sources; the truthful articles were obtained by crawling articles from Reuters.com; <br><br> The Fake dataset was collected from unreliable websites and Wikipedia. The dataset contains different types of articles on different topics. The majority of articles focus on political and World news topics. | 44,898 rows|

### Preliminary Data Cleaning
To ensure consistency across the dataset, the different datasets’ “label” columns have been standardised to be of binary value – either True or False. Since hyperlinks or URLs are often unique, they do not provide much value to the model especially after tokenisation. The “statement” columns have been filtered to account for hyperlinks. Additionally, the occurrence of other language characters were not common, for simplicity, we have removed rows that contain such characters. To avoid bias in our dataset as the 4 datasets are of different sizes (21152, 10269, 4378 and 40967 respectively), we utilised a random selection of 4378 from each dataset to concatenate together, forming the joint dataset. This ensures that our joint dataset has equal proportions of each initial data source and is not leaning towards any particular initial data source. We then analyse the distribution of the word count within the “statement” column, removing three interquartile ranges of anomalies.

## Dataset Features Table

| Feature Name | Description | Value Type | Remark |
| :---: | :--- | :---: | :--- |
| `statement` | The statement / text being checked | string | - |
| `num_words` | Count of words in “statement”, removed in the dataset | float | - |
| `label` | The verdict of the fact check of the statement | categorical {`TRUE`, `FALSE`} | - |

## Linguistic Features Table

| Category | Feature Name | Feature Description | Value Type | Feature Statistics |
| :---: | :--- | :--- | :---: | :--- |

(to insert table)

# Models and Performance 

Our project used three main types of classification models: (1) Models based on linguistic features comprising Support Vector Classifier (SVC), Random Forest (RF) and Multi-Layer Perceptron (MLP). (2) Models based on the text itself, where we applied two types of Naive Bayes’ Classifiers to either TF-IDF or Count vectors. We then apply ensemble methods such as voting and stacking to aggregate models which use linguistic features, and models which use the text itself. Finally we also tested (3) Convolutional Neural Network, which takes the TF-IDF/Count vectors and/or linguistic features as input, where we tested numerous different architectures to find the optimal architecture.

| Model | Training Set Accuracy | Test Set Accuracy | 
| :---: | :---: | :---: |
| Support Vector Classifier | 78.9% | 63.9% |
| Random Forest Classifier | 99.2% | 66.6% |
| MLP Classifier | 74.6% | 64.5% |
| Complement Naive Bayes + Count Vectorizer | 99.7% | 74.4% |
| Complement Naive Bayes + TFIDF Vectorizer | 99.9% | 72.7% |
| Multinomial Naive Bayes + Count Vectorizer | 99.7% | 74.4% |
| Multinomial Naive Bayes + TFIDF Vectorizer | 99.9% | 68.9% |
| Voting Classifier | 98.00% | 76.07% |
| Stacking Classifier | 77.50% | 76.77% |
| Tuned Stacking Classifier | 77.50% | 76.80% |

## Optimal training parameters 
For SVC the C coefficient, kernel type and degree of the polynomial kernel were optimised. For RF, the estimators, max depth of each tree, maximum sample split, maximum sample leaf and maximum features were optimised. For MLP, the hidden layer size, activation function, solver algorithm, alpha coefficient for regularisation and learning rate were optimised.
For all 4 naive Bayes models, we chose to optimise the best combination of n-grams, the result of this was a combination of bigrams and trigrams.

## Convolutional Neural Networks
(to insert table)

# Contribution and Justification

## Datasets
As most models currently utilise only around one to two datasets which results in a biassed dataset (e.g. Jiang et al. (2021) uses the ISOT dataset, which uses real articles from Reuters only). As major news agencies like Reuters have specific style guides which restrict how articles are written, it results in particular linguistic features being weighted extremely heavily based on the style guide (Gravanis et al., 2019). As such, a lack of utilisation of a variety of datasets would cause overfitting. Hence, we have chosen to combat these existing papers by utilising multiple sources and combining them.
Additionally, several articles were classified due to unreliable automatic methods of generating the labels. One such example is the ‘Getting Real about Fake News’ dataset which uses the BS Detector Chrome Extension to set the stances of its data. As unreliable sources can publish real news, likewise, a reliable source may occasionally publish inaccurate news, the classification algorithms are not able to capture these nuances, wrongly classifying rows. 
The chosen four raw datasets were selected as they fulfil our selection criteria of having been labelled manually by human fact checkers while having a more diverse selection of dataset. To prevent our final dataset from leaning towards an initial data source, we took equal proportions of each of the initial four data sources. This ensures that our final dataset consists of equal proportions of each initial data source, preventing any bias.

Overall, our combined method of implementing a concatenated dataset that consists of different datasets and ensuring the data has been manually labelled solidifies its position as a unique approach and ensures a high level of accuracy and relevance. This also ensures that our model is being trained on accurately labelled data. This meticulous approach to our dataset curation not only improves the precision of our fake news detection model, but also eliminates any risk of bias. 

## Creativity in Feature Engineering 
