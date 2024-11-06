# Elevator Pitch:

Breast cancer is a critical health concern, and early detection significantly increases survival rates. This project leverages advanced machine learning and neural network algorithms to improve the accuracy of diagnosing breast cancer at early stages. Using the Wisconsin Breast Cancer Dataset, we assess the power of these algorithms to classify tumors as benign or malignant. Our approach not only aims to enhance detection accuracy but also demonstrates how these technologies can support healthcare providers in diagnosing and managing cancer more effectively. Ultimately, this research contributes to the development of fast, reliable diagnostic tools, empowering earlier intervention and improving patient outcomes in breast cancer care.

# Breast Cancer Detection using Neural Networks and Machine Learning

1. Introduction

Breast cancer in this generation is one of the most alerted health concerns amongst women globally hence, early detection is crucial to adequate prevention and treatment. The earlier breast cancer is found, the better chace of survival as the preventive measures can be taken if the breast cancer is found before it starting to cause major damage to patients. Machine learning algorithms have come wide ways and now, they enable us to make even better predictions on earlier stages of breast cancer which have ultimately saved numerous lives. The American Cancer Association (2021) states that through this strong algorithm, they can detect early breast cancer and intervene before symptoms appear. As far as the medical field is concerned, machine-learning and neural-network-based algorithms have been making a major difference as it has been able to detect various disease accuracy and faster than any human being. These tools minimize the burden on healthcare providers and help patients to start preventive measures early. 

This report presents different machine learning and neural network methods to identify with high accuracy whether breast cancer is benign or malignant. Benign is used in non-cancerous conditions and malignant refers to cancerous condition. Classification uses the Wisconsin Breast Cancer Dataset for clustering, available from UCI Machine Learning (Dua & Graff, 2019). This report will use the dataset to answer how these technologies can be leveraged in practice with computer algorithms and simulate approaches that are more broadly applicable to classify a diagnosis of medical conditions using data obtained therein. In this report, we demonstrate the power of machine learning and neural networks for breast cancer detection and classification using analysis and experimentation. 

2. Literature Review

There have been great advancements made in consideration of utilizing machine learning techniques to solve medical-related problems which have helped in medical research and clinical practices over many years. Sidey-Gibbons & Sidey-Gibbons 2019 is not the only research group following this approach of using machine learning in medical contexts, yet attention on both a conceptual understanding as well as implementing it. In this larger framework is the work of Treu et al. which deals with building predictive algorithms for cancer diagnosis based on a dataset containing samples of both breast cancer and non-cancer records. The authors developed three models, namely regularized General Linear Model Regression, SVM, and Artificial Neural Networks which are single-layered. They train-and-tested-train-thusly their models by doing a cut of the data through an evaluation & validation sample.

Tanaka & Aranha (2019) also do similar work of experimenting with Generative Adversarial Networks to generate data in their research paper which can give us insight, into how it is useful for training data learning. By replacing the example generation mechanisms with a novel, unsupervised method of GANs (if appropriate for the problem at hand), new techniques like SMOTE or ADASYN that over-sample minority classes and preserve privacy. Their work tested the strength of a GAN by evaluating benchmark datasets with different network architectures. Most noteworthy of all, the recall measure was used to assess how well a model not only performed but was able to detect positive instances equally. Two key rationales for generating synthetic datasets are cited in this article by the authors. Instead, it deals with imbalanced datasets as found in credit card fraud detection or medical diagnosis where minority classes might be rare. Such an approach lets the classifier learn minority classes more effectively and avoid overfitting since it comes up with new augmentation data to be generated out of such underrepresented instances. That means that some of your data might be becoming non-compliant with this new regulation. 

Chae & Wilke 2019 also investigate the subset selection of step sizes in the mini-batch sub-sampling (MBSS) for neural network training. You can run the code in two modes: - static mode (where you update mini-batch only after changing the direction of the search). In this study, the second-order quadratic line search corrections are used to ascertain how well that function and derivative information help in constructing these approximations for dynamic MBSS loss functions. Experimented with many neural network tasks, and found that carefully selecting enforced information leads to up-to several orders of magnitude improvements in prediction accuracy in step sizes. These results provide critical insights into a trade-off between bias and variance in MBSS as well as plausible settings of step size dependency that are necessary for defining optimal parameter selection, which is indispensable to the margin (i.e. fastest learning) NNs.

Van Looveren and Klaise (2019) similarly propose a fast, model-agnostic approach to finding interpretable counterfactual explanations of classifier predictions by using class prototypes. The approach is built on top of either a prototype from an encoder or class-specific k-d trees to speed up the process. The approach was tested on image (MNIST) and tabular (Breast Cancer Wisconsin Diagnostic). Counterfactuals: Desire for Interpretability Quyen Luu, Hai Dang, and Shuo Wang tackle the problem of balancing sparsity vs Training desired characteristic notes with interpretability in counterfactual instances. They point out that sparse perturbations can be closer to the underlying manifold when looking at data as a whole, but may not be aligned with one of the two subset classes for which you wish your counterfactual example. The proposed method addresses this issue by including class prototypes in the loss function, avoiding computational bottlenecks related to numeric gradient computations for black-box models. This work is a major step forward in being able to produce human-interpretable counterfactual explanations and provides an interpretable framework that can be used with different data sets and model types. 

3. Problem Statement and Task Performed

3.1. Problem Definition
In the current context many of the women have fallen victim to breast cancer, since breast is a very private part many of the patient suffering from such problems never show up early for checkup which increases the risk of them causing serious damage to themselves in the near future. Likewise, for those that do go to check might be miss judged as their breast cancer has not developed to a significant level. For example, is a women might have breast cancer but in a very small proportionate and if the doctors make a mis judgment as labels her as non-cancerous then she might not start any preventive measures risking her life. In such cases a need of accurately identifying cancer is much in need. 

3.2	Tasks Performed

1.	Basic Exploration:

![1](https://github.com/user-attachments/assets/7821723b-6393-4e1e-b012-6e6b2188476d)

![2](https://github.com/user-attachments/assets/cc5bde81-c0c3-48f0-8109-fdaf87d16970)

2.	Outliers Identification: 
Now to identify the outlies, we take the help of box plot and histogram. For all of the 31 columns, outliers were visualized using box plot and bar graph. Following figure shows the visualization for outliers.

![3](https://github.com/user-attachments/assets/45f27244-cb92-4f40-ab13-14d1cd220fa1)

3.	Correlation Matrix: 
Now to end the exploration phase, a heat map has been generated for all the 32 columns to check the correlation of each of the columns with one another. This is specially used to check the dependencies of 32 columns with the target column. The following figure shows the heat map.

![4](https://github.com/user-attachments/assets/575a14fa-6422-4ac7-9478-78af04faeb48)

4.	Handling Outliers: 
Now since the outliers were identified using the interquartile ranger method, those records that consisted of outliers were dripped form the data frame. This method ensures that only reasonable records or values are present inside the data frame which ensure more accuracy during the time of building model. Further, after dropping the outliers form the data frame the following figures shows the information about the outlier free data frame. There are only 398 records which was initially 569. Further, the comparison between following and above figure shows that many of the data were dropped due to them identifying as outliers.

![5](https://github.com/user-attachments/assets/e277218b-0535-4368-9d57-d4f4ebfcd6fc)

5.	KMeans Clustering and Dimensionality Reduction
First we keep the feature dataset into a KMeans clustering algorithms followed by Principal Component Analysis (PCA) to reduce dimensionality along with visualizing the data into two dimensions. Further, this is done to understand the distribution and structure of the data which is also shown in terms of scatter plot by the following figure.

![6](https://github.com/user-attachments/assets/a635545a-5416-4519-926e-15c7dd7ef292)

6.	Model Evaluation: 
As shown by the following figure, a line graph for model accuracy and loss between the train and validation dataset. The blue line represents the train and orange line represents validation dataset. For the accuracy the train dataset gradually increased and reached the highest around 100 epochs whereas for the validation dataset the accuracy remained constant at around 50 epochs. Likewise for the loss same kind of patterns in downward trend can be observed for both training and validation dataset.

![7](https://github.com/user-attachments/assets/a8afadac-587e-4f6a-8e64-6cf834278710)

7. Confusion Matrix
Now the model will be evaluated based on the testing data by using the prediction model to the testing data and generating a confusion matrix which provides a report on accuracy, precision, recall, and F1-score which is shown by the following figure. 

![8](https://github.com/user-attachments/assets/5c1d5a35-b58c-48a2-bd76-358496801db6)

7.	Discussion of Findings
The dataset consisted of 569 rows and 33 columns out of which ‘diagnosis’ is a  target column and the rest were feature columns. With initial the data exploration, the ‘unnamed:32’ column was found to be completely empty which was dropped for better accuracy.. Hence, out of the 30 columns 29 of them were found to have outliers after. Now the outliers were removed and the new record was 398 rows. This 398 rows had no outliers, as a result, it made the dataset more accurate. Similarly, with the help of correlation matrix the understating of relation between features and target was enhanced as the correlation varied for various feature. Finally, the neural network model was prepared with three hidden layers and tow dropout layer in order to prevent the model in memorizing certain patterns form the training dataset and predicting instead of generalizing. With the analysis of training and validation accuracy and loss through figure 27 and 28 the model performed exceptionally good where it generalized and signs of overfitting were not very prominent. To be precise the model achieved 98.6% and 100% accuracy with the training and validation respectively dataset in over 200 epochs. Likewise, the final evaluation also showed an accuracy of 93% with the unseen dataset. 




