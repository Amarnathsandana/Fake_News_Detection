import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import nltk
import numpy as np
import re
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# File path to the dataset
file_path = '/content/drive/MyDrive/Dissertation Project/modified_news_dataset.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Drop the 'title' column if it exists
data = data.drop(columns=['title'], errors='ignore')

# Check for and handle missing values
print("Missing values before filling:")
print(data.isnull().sum())
data = data.fillna('')

# Define a function for text preprocessing using lemmatization
def preprocess_text_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation and numbers
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return ' '.join(tokens)

# Apply preprocessing to the text column using lemmatization
data['text'] = data['text'].apply(preprocess_text_lemmatizer)

# Display the first few rows of the preprocessed data
print("Preprocessed data:")
print(data.head())

# Separate real and fake news for visualization
real_news = data[data['label'] == 1]
fake_news = data[data['label'] == 0]

# Function to generate and display word clouds
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, size=20)
    plt.axis('off')
    plt.show()

# Generate word clouds for real and fake news
generate_wordcloud(' '.join(real_news['text']), 'Real News')
generate_wordcloud(' '.join(fake_news['text']), 'Fake News')

# Function to get the top N words from the corpus
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Function to plot top N words
def plot_top_words(words_freq, title):
    words = [word for word, freq in words_freq]
    freqs = [freq for word, freq in words_freq]
    sns.barplot(x=freqs, y=words)
    plt.title(title)
    plt.show()

# Plot top 20 words
top_words = get_top_n_words(data['text'], n=20)
plot_top_words(top_words, 'Top 20 Words in News Articles')

# Split the data into training and testing sets for traditional models
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train and evaluate Naive Bayes model
print("Naive Bayes Model:")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb, target_names=['Fake', 'Real'])

print(f'Accuracy: {accuracy_nb:.4f}')
print('Confusion Matrix:')
print(conf_matrix_nb)
print('Classification Report:')
print(report_nb)

# Plot Confusion Matrix for Naive Bayes
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

# Train and evaluate K-Nearest Neighbors model
print("K-Nearest Neighbors Model:")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_vec, y_train)
y_pred_knn = knn_model.predict(X_test_vec)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn, target_names=['Fake', 'Real'])

print(f'Accuracy: {accuracy_knn:.4f}')
print('Confusion Matrix:')
print(conf_matrix_knn)
print('Classification Report:')
print(report_knn)

# Plot Confusion Matrix for K-Nearest Neighbors
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - K-Nearest Neighbors')
plt.show()

# Integrate LSTM model

# Text preprocessing function using stemming
ps = PorterStemmer()
def preprocess_text_stemmer(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    # Tokenize and remove stopwords, then stem words
    text = [ps.stem(word) for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(text)

# Apply preprocessing to the text data using stemming
data['processed_text'] = data['text'].apply(preprocess_text_stemmer)

# Define parameters for embedding
voc_size = 5000
sent_length = 20
embedding_vector_features = 40

# One-hot encode the text data
onehot_repr = [one_hot(words, voc_size) for words in data['processed_text']]

# Pad sequences to ensure uniform input length
embedded_docs = pad_sequences(onehot_repr, padding='post', maxlen=sent_length)

# Prepare the final input and output arrays
X_final = np.array(embedded_docs)
y_final = np.array(data['label'])

# Split the data into training and testing sets for the LSTM model
X_train_embed, X_test_embed, Y_train_embed, Y_test_embed = train_test_split(X_final, y_final, test_size=0.33, random_state=27)

# Build the LSTM model
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
print("LSTM Model Summary:")
print(model.summary())

# Train the model
model.fit(X_train_embed, Y_train_embed, validation_data=(X_test_embed, Y_test_embed), epochs=10, batch_size=64)

# Predict on the test set
Y_pred_embed = (model.predict(X_test_embed) > 0.5).astype(int)

# Evaluate the model's performance
accuracy_lstm = accuracy_score(Y_test_embed, Y_pred_embed)
print(f"Accuracy: {accuracy_lstm * 100:.2f}%")

# Generate confusion matrix and classification report
conf_matrix_embed = confusion_matrix(Y_train_embed, Y_pred_embed)
classification_report_embed = classification_report(Y_train_embed, Y_pred_embed, target_names=['Fake', 'Real'])

print("Classification Report:\n", classification_report_embed)
print("Confusion Matrix:\n", conf_matrix_embed)

# Plot Confusion Matrix for LSTM Model
sns.heatmap(conf_matrix_embed, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - LSTM Model')
plt.show()

# Display all model accuracies for comparison
print("Model Comparison:")
print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
print(f"K-Nearest Neighbors Accuracy: {accuracy_knn:.4f}")
print(f"LSTM Model Accuracy: {accuracy_lstm:.4f}")
