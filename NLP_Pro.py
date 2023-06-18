# import important libraries
import itertools
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style


style.use('ggplot')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve , auc

# ------------------------------------------------------------------
# read data from Text file
negatives = os.listdir("txt_sentoken/neg")
neg_sentence = []

Positives = os.listdir("txt_sentoken/pos")
pos_sentence = []

for path in negatives:
    with open('txt_sentoken/neg/' + path) as file:
        lines = file.readlines()
        merged_line = ' '.join(lines).replace('\n', '')
    neg_sentence.append(merged_line)

for path in Positives:
    with open('txt_sentoken/pos/' + path) as file:
        lines = file.readlines()
        merged_line = ' '.join(lines).replace('\n', '')
    pos_sentence.append(merged_line)

# ------------------------------------------------------------------
# read data in data frames
sentence = pd.DataFrame(pos_sentence)
sentence['target'] = 1
sentence.columns = ['review', 'target']
sentence2 = pd.DataFrame(neg_sentence)
sentence2['target'] = 0
sentence2.columns = ['review', 'target']
sentence = sentence.append(sentence2, ignore_index=True)
sentence = sentence.sample(frac=1, random_state=1, ignore_index=True)

# -----------------------------------------------------------------
# visiualize the data
sns.countplot(x='target', data=sentence)
plt.show()
def plot_confusion_matrix(confusion_matrix, classes):
    # Normalize the confusion matrix
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'  # Format for displaying values inside the confusion matrix
    thresh = confusion_matrix.max() / 2.0
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_classification_report(report,name):
    class_names = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'

    # Extract precision, recall, and F1-score values
    precision = [report[class_name]['precision'] for class_name in class_names]
    recall = [report[class_name]['recall'] for class_name in class_names]
    f1_score = [report[class_name]['f1-score'] for class_name in class_names]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(len(class_names))
    bar_width = 0.25
    opacity = 0.8

    # Plot the precision, recall, and F1-score bars
    rects1 = plt.bar(index, precision, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Precision')

    rects2 = plt.bar(index + bar_width, recall, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Recall')

    rects3 = plt.bar(index + 2*bar_width, f1_score, bar_width,
                     alpha=opacity,
                     color='r',
                     label='F1-score')

    # Set the axis labels and ticks
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Classification Report '+name)
    plt.xticks(index + bar_width, class_names)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------
# preprocessing
updated_stop_words = [word for word in stop_words if not word.__contains__("n't")]


def preprocessing(text):
    # Segment sentences
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    # Process each sentence
    for sent in sentences:
        s = sent
        # s = s.translate(str.maketrans('', '', string.punctuation))
        s = s.replace('[^a-zA-Z0-9\s]', '')
        s = s.lower()
        s = word_tokenize(s)
        # s = [w for w in s if not w in updated_stop_words]
        lemmatizer = WordNetLemmatizer()
        s = [lemmatizer.lemmatize(w) for w in s]
        s = ' '.join(s)
        cleaned_sentences.append(s)
    cleaned_text = ' '.join(cleaned_sentences)
    return cleaned_text


sentence['review'] = sentence['review'].apply(preprocessing)

# -----------------------------------------------------------------
# Factorizing data
Y = sentence['target']
vect = TfidfVectorizer()
X = vect.fit_transform(sentence['review'])
x_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
# -----------------------------------------------------------------
# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# -----------------------------------------------------------------
# MultinomialNB Model
MNP = MultinomialNB()
MNP.fit(x_train, y_train)

# -----------------------------------------------------------------
# SVM Model
SVC = LinearSVC()
SVC.fit(x_train, y_train)

# -----------------------------------------------------------------
# Evaluation of model Logistic Regression
y_predlog = logreg.predict(x_val)
# Calculate accuracy
accuracy = accuracy_score(y_val, y_predlog)
print('\n', '\n', '\n', 'Evaluation OF Logistic Regression MODEL''\n')
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_val, y_predlog)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_val, y_predlog)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_val, y_predlog)
print("F1-Score:", f1)

# Generate classification report
report = classification_report(y_val, y_predlog)
print("Classification Report:")
print(report)
#plot_classification_report(report,'LogisticRegression')

# Generate confusion matrix
cm = confusion_matrix(y_val, y_predlog)
print("Confusion Matrix:")
print(cm)

# -----------------------------------------------------------------
# Evaluation of MultinomialNB Model
y_predmnp = MNP.predict(x_val)
# Calculate accuracy
accuracy = accuracy_score(y_val, y_predmnp)
print('\n', '\n', '\n', 'Evaluation OF MultinomialNB MODEL' '\n')
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_val, y_predmnp)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_val, y_predmnp)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_val, y_predmnp)
print("F1-Score:", f1)

# Generate classification report
report = classification_report(y_val, y_predmnp)
print("Classification Report:")
print(report)
#plot_classification_report(report, 'MultinomialNB')

# Generate confusion matrix
cm = confusion_matrix(y_val, y_predmnp)
print("Confusion Matrix:")
print(cm)

# -----------------------------------------------------------------
# Evaluation of SVM Model
y_predsvc = SVC.predict(x_val)
# Calculate accuracy
accuracy = accuracy_score(y_val, y_predsvc)
print('\n', '\n', '\n', 'Evaluation OF SVM MODEL''\n')
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_val, y_predsvc)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_val, y_predsvc)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_val, y_predsvc)
print("F1-Score:", f1)

# Generate classification report
report = classification_report(y_val, y_predsvc)
print("Classification Report:")
print(report)
#plot_classification_report(report, 'SVM')
# Generate confusion matrix
cm = confusion_matrix(y_val, y_predsvc)
print("Confusion Matrix:")
print(cm)

# -----------------------------------------------------------------
# Testing of SVM Model
y_pred = SVC.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

#  Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
#plot_classification_report(report, 'Test of SVM')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['Positive', 'Negative']
plot_confusion_matrix(cm, class_names)
plt.show()


# Calculate ROC curve and AUC (if applicable)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# Plot ROC curve (if applicable)
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()