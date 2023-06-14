# CSE256_Project
Question Answering Project

All the data files are present here: https://drive.google.com/drive/folders/1MgduoD_7oNH6QNul98TAkxV0xHFoc6tm?usp=sharing

Following is a code description for each of the file present in this repo:
1) BERT_Retriever.ipynb: Retriever module that used embedding from the BERT models and finds cosine similarity of the questions with all the paragrphs in the document. It requires the data files 'test_data.xml' and 'data_withNeg1k-5Cols.csv' available in the above google drive folder.
2) Baseline_QandA.ipynb: Used to evaluate the basline model performance on a single xml file. It requires the data files 'test_data.xml'.
3) Preprocessing.ipynb: Code for web scraping and span detection on a single xml file. It requires the data files 'test_data.xml'.
4) REALM_Retriever.ipynb: Retriever module that uses Realm_Scorer and Realm_Retriever models and finds similarity of the questions with all the paragrphs in the document. It requires the data file 'data_withNeg1k-5Cols.csv'.
5) bleu_baseline.py: Used to obtain baseline bleu scores on the entire data. It requires the data files 'test_200-5Cols.csv' and 'baseline_results_400New.csv'.
6) t5_finetuning_MedQuAD.py: Code used to train the t5 model on the MedQuad dataset. It requires the data file 'data_2k_shuffled.csv'.
7) testModel.py: Code used to evaluate the performance of the saved and trained t5 model on the test data. It requires the data file 'data_withNeg2k_realm.csv'
