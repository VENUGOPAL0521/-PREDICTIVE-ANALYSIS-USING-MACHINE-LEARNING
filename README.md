# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: EMANDI VENUGOPAL NARAYANA

*INTERN ID*: CT04DN384

*DOMAIN*: DATA ANALYSIS

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

In today's data-driven world, businesses thrive on their ability to make informed decisions backed by data analytics. One crucial area where machine learning plays a significant role is predictive analysis—forecasting outcomes using historical data. This project demonstrates the practical application of machine learning techniques to predict customer churn, which is the propensity of customers to leave a company's service. Customer churn prediction is vital for companies, especially those in the subscription or telecom industry, as retaining customers is often more cost-effective than acquiring new ones.
For this project, we used the Telco Customer Churn dataset, a popular open-source dataset that includes various features such as customer demographics, account information, service subscription details, and churn status (Yes/No). The primary objective is to build a model that accurately classifies whether a customer will churn or stay, enabling proactive customer retention strategies.
The first phase of the project involved data preprocessing, a critical step in preparing raw data for machine learning models. We handled missing values, converted data types (e.g., total charges from string to numeric), and encoded categorical variables using label encoding and one-hot encoding methods. Feature scaling was applied to numerical columns using StandardScaler to ensure that all features contribute equally to model training.
Next, we performed feature selection to identify which attributes were most relevant for predicting churn. A combination of domain knowledge, correlation analysis, and Recursive Feature Elimination (RFE) helped us reduce dimensionality and improve model efficiency. Features like tenure, monthly charges, contract type, and payment method stood out as strong indicators of churn.
We experimented with multiple classification algorithms such as Logistic Regression, Random Forest, XGBoost, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN). Among these, Random Forest was selected as the final model because it provided a good balance of accuracy and interpretability and handled class imbalance effectively. We fine-tuned this model using GridSearchCV, optimizing hyperparameters like the number of trees (n_estimators), tree depth (max_depth), and minimum samples required to split a node (min_samples_split).
In conclusion, this project illustrates the power of machine learning in solving real-world business problems. It covers the entire data science workflow—from preprocessing and feature engineering to model training and evaluation. The deliverables include a Jupyter notebook, Python script, and a project report. Such predictive models empower organizations to make strategic decisions that reduce churn and improve customer satisfaction.
