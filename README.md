# NLP
Sentiment Analysis for Drug Effectiveness, Satisfaction and Side Effects Using Natural Language Processing (NLP) and Deep Learning Methods. Project completed for DSCI 691 Natural Language Processing with Deep Learning on 6/12/2024 at Drexel University.

Abstract: 

In this project, we develop and assess sentiment analysis models employing both traditional Natural Language Processing (NLP) and deep learning methodologies to enhance the identification of sentiments associated with drug effectiveness,satisfaction and side effects across different sources. Sentiment analysis, a critical task in NLP, is employed to understand emotional distinctions in textual data, thereby uncovering attitudes, opinions, and emotions. We incorporate data from two primary sources: the Drugs.com Reviews Dataset and the WebMD Drug Reviews Dataset, both of which are openly available through kaggle. We discern and classify sentiments pertaining to drug effectiveness, satisfaction and side effects from textual reviews using traditional NLP methods with promising results. Leveraging TensorFlow, a deep learning framework, and advanced neural networks using Recurrent Neural Network (RNN) Gated Recurrent Network (GRU) and Long Short Term Memory (LSTM), our deep learning models exhibit promising performance, demonstrating scalability and efficiency in handling substantial amounts of data without compromising accuracy, thus facilitating potential large-scale deployment. We analyze the models for Cross-data adaptability and the results are also promising as it's key to establish a sentiment analysis model that is not only effective but also generalizable across various medical conditions and drug types. This analysis could help organizations better understand customer sentiment and influence informed decision-making in healthcare.

1. The contents of the Sentiment Analysis .ipynb file are separated into the following sections: 

   Team:
	
	Jimmy Zhang, Thu Tran, and Shawn Oyer

   Introduction and Background:

	Why we selected this topic and what we hope to achieve in the project.

   Project Data:

   	The data we chose to incorporate in the project are contained within two data sources.

   	WebMD Drug Reviews Dataset (https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)

   	Data was acquired from scraping WebMD (https://www.webmd.com/drugs/2/index) and compiled into a .csv (Comma-separated values) containing 362,806 	rows x 12 columns.
   	The dataset provides user reviews on specific drugs along with related conditions, side effects, age, sex, and ratings reflecting overall patient   
   	satisfaction.

   	Description of Attribution:
		Drug (categorical): name of drug
		DrugId (numerical): drug id
		Condition (categorical): name of condition
		Reviews (text): patient review
		Sides (text): side effects associated with drug (if any)
		EaseOfUse (numerical): 5 star rating
		Effectiveness (numerical): 5 star rating
		Satisfaction (numerical): 5 star rating
		Date (date): date of review entry
		UsefulCount (numerical): number of users who found review useful
		Age (numerical): age group range of user
		Sex (categorical): gender of user

   	Data was acquired from crawling online pharmaceutical review sites such as Drugs.com (https://www.drugs.com/) and compiled into two .tsv (Tab-	separated values) containing a total of 215,063 rows x 7 columns. They are partitioned by training (161,297, 75%) and testing (53,766, 25%) 	respectively.
   	The dataset provides patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient  
   	satisfaction.

   	Description of Attribution:
		id (integer): drug id
		DrugName (categorical): drug name
		Condition (categorical): name of condition
		Review (text): patient review
		Rating (numerical): 10 star rating
		Date (text): review date
		UsefulCount (numerical): number of users who found review useful

   Utilities:

	Import packages and modules

   EDA and Pre-Processing:
	
	Accessing the data, acquiring information about the data, cleaning the data by dropping null values, create new columns for the binary 	classification, plot bar plots, histograms, correlation matrices, and word clouds.

   In-Domain Sentiment Analysis:

	Traditional NLP Methods:

	Drug Effectiveness and Satisfaction Analysis using Logistic Regression:

	Drug reviews from both datasets were processed by converting to lower case, removed punctuation and number, tokenized on spaces, and returned as a 	new column called clean_text. The clean_text column was run through the count vectorizer to extract n-gram features from 1-3 with a threshold of 	0.95. A new binary column was created as follows: 0 for rating <=4, 1 for rating between 4 and 7, and 2 for rating >=7. The clean_text column was 	then used as the training/testing input and the newly created binary rating column was used for what we were trying to predict. The data was run 	through a pipeline, and a grid search with 5-fold cross-validation was conducted using Logistic Regression with a class_weight of balanced to 	avoid class imbalance and potential overfitting of the model. The scoring metrics used were Accuracy and Cohen's Kappa 	Score. We included a chart 	displaying the prediction and test models.

	Side Effect Analysis using Linear Regression:

	Drug reviews from both datasets were processed by converting to lower case, removed punctuation and number, tokenized on spaces, and returned as a 	new column called SideRating. The SideRating column was run through a counter function to sum all of the counts in the column. A new binary column 	was created as follows: 0 for No side effects, 1 for Mild/Moderate side effects, and 2 for Severe/Extremely Severe side effects. All of the 	columns minus the SideRating were used as the training/testing input and the SideRating column was used for what we were trying to predict. The 	data was run through a pipeline using a standard scaler and median imputer strategy and a Linear Regression model was fitted. The scoring metrics 	used were Accuracy, RMSE and Cohen's Kappa Score. We included the output classification report and a chart with the top features by coefficient 	magnitude.

	Deep Learning Methods:

	Drug Effectiveness and Satisfaction Analysis using Recurrent Neural Networks (RNN) Gated Recurrent Unit (GRU):

	Drug reviews from both datasets were processed by tokenizing the reviews, converting text to sequences of integers, and pad sequences. The 	SideRating column was run through a counter function to sum all of the counts in the column. The padding sequences were used as the 	training/testing input and the rating binary column was used for what we were trying to predict. Unique class labels were used to calculate 	balanced class weights since the data was imbalanced. The model was defined with dropout layers and compiled with loss as 	sparse_categorical_crossentropy, optimizer as adam, and accuracy metrics. We plotted the model accuracy and model loss, in addition to the 	classification report and confusion matrices. We also created a function that tests the model on arbitrary reviews to determine how accurate the 	model was in predicting sentiment. 

    Cross-Data Sentiment Analysis:

	Streamlined both datasets by dropping unnecessary columns and renaming the remaining ones. We utilize linear regression to create a cross-data 	model in two cases. In Case 1, we compare the Drug.com train set with the WebMD test set and the Linear Regression (LR) model predictions, 	achieving a train accuracy of 99.99%, test accuracy of 84.08%, RMSE score of 0.307, and Cohen's Kappa score of 0.392. Despite the model 	overfitting the train set, the test accuracy was higher compared to in-domain sentiment analysis, and the Kappa score exceeds the document 	reference. The graph shows significant fluctuations in both the WebMD test set (blue line) and the LR model predictions (orange line), indicating 	high variability in ratings and facilitating the assessment of the model's alignment with actual test ratings. In Case 2, we compare the Drug.com 	test set with the WebMD train set and the LR model predictions. The graph similarly highlights considerable fluctuations, and the performance 	metrics are consistent with Case 1, underscoring the model's predictive performance and variability.

    Results:

	Tables to visualize comparison of all models and explanations of those comparisons.

    Discussion:

	Discussion on how the results were interpreted.

    Conclusion:

	Overall, this study offers valuable insights into customer sentiments surrounding drug effectiveness, satisfaction and side effects. The findings 	presented here hold significant implications for healthcare organizations and decision-makers, providing a foundation for informed decision-making 	processes and enhancing understanding of patient experiences and perspectives.

    References:

	List of References used in the project.

2. Stakeholders and Use

   Healthcare Organizations, drug makers, drug consumers, etc can use this analysis to better understand customer sentiment and influence informed 
   decision-making in healthcare.
  
3. Challenges/Limitations

   Logistic Regression and Linear Regression models were severely overfitted (Ran out of time to properly reduce overfitting).

   GPUs had to be used by purchasing google collab pro to process over 15 million parameters as the CPUs crashed every time.

   Traditional NLP and Deep Learning Model refinement, feature engineering and more in-depth processing of textual reviews needed to achieve better   
   overall results.

   Continue research by performing Cross-Domain Sentiment Analysis by evaluating the transferability of models among medical domains, (i.e. conditions 
   such as Depression vs Anxiety) 

4. Using the Report

   The extension of the script is .ipynb so it can be accessed and run within jupyter notebook and exported as a .py to export into any Python IDE.

5. Contributors and Contact List

   Jimmy Zhang - Drexel University Doctoral Student, jz876@drexel.edu

   Shawn Oyer - Drexel University Gradate Student, sbo33@drexel.edu

   Thu Tran - Drexel University Graduate Student, tt537@drexel.edu

6. License Use

   Creative Commons (CCO) v1.0-v4.0 Universal- 

   No Copyright
   The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide  
   under copyright law, including all related and neighboring rights, to the extent allowed by law.
   You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission. See Other Information below.

   Other Information
   In no way are the patent or trademark rights of any person affected by CC0, nor are the rights that other persons may have in the work or in how the   
   work is used, such as publicity or privacy rights.
   Unless expressly stated otherwise, the person who associated a work with this deed makes no warranties about the work, and disclaims liability for all   
   uses of the work, to the fullest extent permitted by applicable law.
   When using or citing the work, you should not imply endorsement by the author or the affirmer.

7. Sources

	Gräßer, F., et al. “Drug Reviews (Drugs.com).” UCI Machine Learning Repository, 03 10 2018, 	https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com. Accessed 4 June 2024.

	Gräßer, F., et al. “Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.” Digital Health, vol. 18, no. 1, 	2018, pp. 1-5. Data Summarization, https://kdd.cs.ksu.edu/Publications/Student/kallumadi2018aspect.pdf. Accessed 1 May 2024.

	Luay, Muhammad. “Sentiment Analysis using Recurrent Neural Network(RNN),Long Short Term Memory(LSTM) and….” Medium, 21 September 2023, 	https://medium.com/@muhammadluay45/sentiment-analysis-using-recurrent-neural-network-rnn-long-short-term-memory-lstm-and-38d6e670173f. Accessed 10 	June 2024.

	Pedamallu, Hemanth. “RNN vs GRU vs LSTM. In this post, I will make you go… | by Hemanth Pedamallu | Analytics Vidhya.” Medium, 14 November 2020, 	https://medium.com/analytics-vidhya/rnn-vs-gru-vs-lstm-863b0b7b1573. Accessed 11 June 2024.

	Rohan, Harode. “WebMD Drug Reviews Dataset.” Kaggle, 01 06 2020, https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset. 	Accessed 4 June 2024.

	Zargar, Sakib A. “Introduction to Sequence Learning Models: RNN, LSTM, GRU Sakib Ashraf Zargar Department of Mechanical and Aerospace 	Engineering.” ResearchGate, 2021, https://www.researchgate.net/profile/Sakib-Zargar-2/publication/350950396	_Introduction_to_Sequence_Learning_Models_RNN_LSTM_GRU/links/607b41c0907dcf667ba83ade/Introduction-to-Sequence-Learning-Models-RNN-LSTM-GRU.pdf. 	Accessed 11 June 2024.
