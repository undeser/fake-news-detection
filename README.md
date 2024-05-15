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
