import os
import PyPDF2
import nltk
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import re

#Baixa os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')

#Função para extrair texto de um arquivo PDF
def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f) #Lê o arquivo PDF
        text = ""
        for page in range(len(reader.pages)): #Itera por todas as páginas do PDF
            text += reader.pages[page].extract_text() #Extrai o texto de cada página
    return text

#Função para limpar texto usando regex
def clean_text_regex(text):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()) #Remove caracteres especiais e deixa o texto em minúsculas
    return clean_text

#Definição dos diretórios que contêm os PDFs classificados por tipo de texto
directories = {
    'poesia': 'pdfs/poesia/',
    'prosa': 'pdfs/prosa/',
    'jornalismo': 'pdfs/jornalismo/'
}

#Processa os textos e atribui os rótulos correspondentes
texts = [] #Lista para armazenar os textos processados
labels = [] #Lista para armazenar os rótulos (classes)
for label, path in directories.items():
    if os.path.exists(path): #Verifica se o diretório existe
        for file in os.listdir(path): #Itera pelos arquivos do diretório
            if file.endswith('.pdf'): #Filtra apenas arquivos PDF
                text = pdf_to_text(os.path.join(path, file)) #Extrai o texto do PDF
                clean_txt = clean_text_regex(text) #Limpa o texto extraído
                if clean_txt: #Verifica se o texto não está vazio
                    texts.append(clean_txt) #Adiciona o texto limpo à lista
                    labels.append(label) #Adiciona o rótulo correspondente

#Cria a matriz Bag of Words (BoW) usando TF-IDF para representar os textos
vectorizer = TfidfVectorizer() #Inicializa o vetorizador TF-IDF
X = vectorizer.fit_transform(texts) #Transforma os textos em vetores TF-IDF
y = np.array(labels) #Converte os rótulos para um array numpy

#Define os classificadores que serão testados e seus hiperparâmetros
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Neural Network": MLPClassifier(max_iter=500)
}

#Validação cruzada com 10 folds
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) #Configura a validação cruzada estratificada

#Avalia os classificadores com validação cruzada e gera CSVs individuais
for name, clf in classifiers.items():
    print(f"\n=== Validação Cruzada para: {name} ===")

    accuracy_scores = [] #Lista para armazenar as acurácias por fold
    f1_scores = [] #Lista para armazenar os F1-scores por fold

    #Lista para armazenar os dados para o CSV do classificador atual
    csv_data = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        #Divide os dados em treino e teste para o fold atual
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #Treina o modelo
        clf.fit(X_train, y_train)

        #Faz previsões
        y_pred = clf.predict(X_test)

        #Calcula as métricas para o fold atual
        acc = accuracy_score(y_test, y_pred) #Calcula a acurácia
        f1 = f1_score(y_test, y_pred, average='weighted') #Calcula o F1-score ponderado

        #Salva as métricas
        accuracy_scores.append(acc)
        f1_scores.append(f1)

        #Adiciona os dados para o CSV
        csv_data.append({
            "Fold": fold,
            "Accuracy": acc,
            "F1-Score": f1
        })

        #Exibe os resultados do fold atual
        print(f"Fold {fold}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")

    #Calcula e exibe as métricas médias e desvios padrão após os 10 folds
    mean_acc = np.mean(accuracy_scores) #Média da acurácia
    std_acc = np.std(accuracy_scores) #Desvio padrão da acurácia
    mean_f1 = np.mean(f1_scores) #Média do F1-score
    std_f1 = np.std(f1_scores) #Desvio padrão do F1-score

    print(f"\nResultados gerais para {name}:")
    print(f"Accuracy - Média: {mean_acc:.4f}, Desvio Padrão: {std_acc:.4f}")
    print(f"F1-Score - Média: {mean_f1:.4f}, Desvio Padrão: {std_f1:.4f}")

    #Salva os resultados no arquivo CSV individual para o classificador atual
    output_csv_path = f"{name.replace(' ', '_').lower()}_folds.csv"
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Fold", "Accuracy", "F1-Score"])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Os resultados foram salvos no arquivo: {output_csv_path}")
