# Resume NER Prediction Using Bert
## **Introduction**
In today's competitive job market, extracting relevant information from resumes efficiently is crucial for both recruiters and job seekers. Named Entity Recognition (NER) plays a vital role in this process by identifying and classifying key elements such as names, skills, education, and work experience within unstructured text. This project aims to leverage the power of BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art language model, to enhance the accuracy and efficiency of resume parsing. By fine-tuning BERT for the specific task of NER in resumes, we aim to improve the extraction of pertinent information, facilitating better matching between candidates and job opportunities.

## **Installation**

This project is run on Kaggle instead of Google Colab. Before running the code, ensure that the following libraries are pre-installed:

- **Torch**: For deep learning support and model training.
- **Transformers**: For utilizing pre-trained models and tokenizers.
- **NLTK**: For natural language processing tasks and data preprocessing.
- **Sklearn**: For various machine learning utilities and metrics.
- **Tqdm**: For displaying progress bars in loops.
- **Pandas**: For data manipulation and analysis.

To install these libraries, you can use the following command:

```python
!pip install torch transformers nltk scikit-learn tqdm pandas
```

## **Data Sources**

The images and labels for this project were downloaded from [Resume Entities for NER](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner/data). 

The dataset includes 10 categories: Name, College Name, Degree, Graduation Year, Years of Experience, Companies worked at, Designation, Skills, Location, and Email Address.

To evaluate the model the data was split into 85% (187 documents) for training and 15% (33 documents) for validation.

![image](https://github.com/user-attachments/assets/ba036a9f-bbea-4172-b3a5-d09115736850)


## **Training Results**

Despite the small size of the dataset, I was able to achieve a **90% F1 score (weighted)**. This demonstrates the effectiveness of the BERT model in performing Named Entity Recognition (NER) on resumes, even with limited training data.

## **Future Work**

To further enhance performance, creating additional annotated documents using large language models (LLMs) like ChatGPT could be beneficial. This would increase the dataset size and improve the model's ability to generalize, leading to more accurate predictions in real-world applications.
