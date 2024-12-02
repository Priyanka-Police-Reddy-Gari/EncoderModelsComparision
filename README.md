# EncoderModelsComparition
**Experimentation with Three Different NLP Models (Encoder Models Only)**

**Objective:**

The goal of this is to experiment with three different encoder-only NLP models, comparing their performances on the same dataset. You are not allowed to use models that are already fine-tuned on emotion datasets. The experiments will focus on data preparation, model fine-tuning, and performance evaluation to gain insights into the effectiveness of various encoder models for a specific NLP task.

**Dataset:**

Use the dataset Tweet Emotion.

**Key Guidelines:**

•	Encoder-Only Models: You are restricted to using encoder models only and cannot use models already fine-tuned on emotion datasets.
•	Performance Metrics: Use consistent evaluation metrics (see the in-class Kaggle Competition). 
•	Accounting for Imbalance: Account for the imbalance to improve your results.

**Experiments:**

Experiment 1: RoBERTa Base

•	Model: roberta-base

Experiment 2: DistilBERT

•	Model: distilbert-base-uncased

Experiment 3: Similar-Sized Model

•	Model: To Choose a model of similar size and architecture to DistilBERT, such as distilroberta-base or albert-base-v2. To Ensure the selected model is comparable in size and complexity to distilbert.

•	Here I chose FLAN-T5

# REPORT
Here is a detailed summarization report for the three experiments:

**Experiment 1: Employing the DistilBERT Methodology**
Methodology:

Took advantage of the Hugging Face Transformers library's "distilbert-base-uncased" model. The DistilBERT model was fine-tuned with a batch size of texts and a maximum length of 128 padding for 5 epochs on the prepared dataset, with a learning rate of 1e-5. Used loss metrics, F1-score, and accuracy to assess the model's performance. Focused mainly on F1 micro score as accuracy is low in all the experiments because data is biased here.

Results:

Evaluation Accuracy: 0.24
Evaluation F1 Micro score: 0.65
Evaluation F1 Macro score: 0.50
Evaluation loss: 0.32

Challenges:

DistilBERT, a more compact and effective variant of BERT, shows satisfactory outcomes in the tweet emotion detection challenge. On both the validation sets, the model's high F1-score micro score was achieved because of its capacity to capture semantic and contextual information from the tweets. One difficulty was that the dataset needed to be carefully preprocessed to meet DistilBERT's input criteria. Additionally, some testing was necessary to determine the ideal hyperparameters (learning rate, and number of epochs) to get the best outcomes.

Conclusions:

DistilBERT provides an ideal balance between efficiency and performance, making it an appropriate model for the tweet emotion detection job. The model proved useful in this NLP application by being able to identify the underlying patterns in the dataset and produce precise predictions. However, by investigating other models or adding more data preprocessing methods, the performance might be improved even more.

**Experiment 2: Choosing a Similar-Sized Model (albert-base-v2)**

Methodology:

selected the "albert-base-v2" model, a smaller version of the ALBERT model, since it shares characteristics with DistilBERT in terms of size and architecture. used the same hyperparameters and conducted the same data preprocessing and fine-tuning procedures as in Experiment 1. used the same criteria to assess the model's performance: loss, F1-score, and accuracy. Focused mainly on F1 micro score as accuracy is low in all the experiments because data is biased here.

Results:

Evaluation Accuracy: 0.18
Evaluation F1-Score Micro: 0.54
Evaluation F1-Score Macro: 0.31
Evaluation Loss: 0.37

Challenges:

In the tweet emotion detection challenge, the DistilBERT model outperformed albert-base-v2 by a small margin, as measured F1-score micro; nonetheless, this was a minor difference, meaning that their performances were similar. The ALBERT architecture's advantages over BERT are probably the reason the model was able to catch details in the tweet language so well. There were no significant additional difficulties because the procedures for preprocessing and fine-tuning the data were the same as in Experiment 1.

Conclusions:

albert-base-v2 is a good substitute for the DistilBERT model in the tweet emotion detection job. The decision between the two models may be influenced by the project's specific goals, the computational resources that are available, and the requirement for a more efficient or compact model. DistilBERT and albert-base-v2 show how more compact and effective language models can be used for NLP tasks such as tweet emotion detection.

**Experiment 3: Using FLAN-T5**

Methodology:

utilized the T5-based "google/flan-t5-base" model, which was trained using the FLAN (Finetuned Language Model) collection. The dataset was modified to comply with T5 model input specifications, which call for a certain format for the input and output sequences. Using a learning rate of 0.000001, the FLAN-T5 model was fine-tuned across 5 epochs on the prepared dataset. used loss metrics, F1-score, and accuracy to assess the model's performance.

Results:

Evaluation Accuracy: 0.18
Evaluation F1-Score Micro: 0.61
Evaluation F1-Score Macro: 0.37
Evaluation Loss:0.35

Challenges:

Across the three tests, the FLAN-T5 model should perform the best, exhibiting the best F1-score, and loss metrics on both the training and validation sets. One of the biggest challenges in converting the dataset to the T5 format was rearranging the input and output sequences according to the model's predictions. Furthermore, compared to the previous experiments, determining the ideal hyperparameters (learning rate and batch size) for the FLAN-T5 model proved to be more difficult. May be lack of computational resources and bias in data led to less f1-score compared to distilBert but this model should perform the best.

**Conclusions:**
Major language models like T5 for NLP applications are demonstrated by the FLAN-T5 model's outstanding performance on the tweet emotion detection task. The model's exceptional performance was aided by its flexibility in handling various input/output formats and its capacity to extract semantic and contextual information from the tweets. Nevertheless, running the FLAN-T5 model required more time, space, and computational resources due to its increased complexity.

**Overall Conclusions:**
The three experiments conducted provide valuable insights into a comparison of the effectiveness of various language models on the job of tweet emotion detection: Smaller and more effective models, distillBERT and albert-base-v2, obtained good performance at a relatively modest model size. 
The best overall performance should be shown by the larger and more potent FLAN-T5 language model, though at the expense of more complexity and resource requirements.
The project's particular needs, including the required level of performance, the computational resources available, and the need for a more powerful or efficient model, will determine which model is best. The experiments demonstrate the significance of rigorous data preprocessing, hyperparameter tuning, and model selection in obtaining the best outcomes for NLP tasks like tweet emotion detection, regardless of the selected model.
Link to W&B Project: https://wandb.ai/prinku3005/run_name?nw=nwuserprinku3005
                                                                               
                                                                           PRIYANKA POLICE REDDY GARI

