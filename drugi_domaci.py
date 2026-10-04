import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_bez_lematizacije(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_sa_lematizacijom(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)
preprocessor_bez_lematizacije = FunctionTransformer(lambda x: x.apply(preprocess_bez_lematizacije))
preprocessor_sa_lematizacijom = FunctionTransformer(lambda x: x.apply(preprocess_sa_lematizacijom))

df = pd.read_csv('train_v2_drcat_02.csv')
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
best_acc = 0
best_model = {}
best_pipeline = 0
rezultati = []

#logisticka regresija sa count vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', CountVectorizer()),
    ('model', LogisticRegression(max_iter=1000, C=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Logisticka regresija sa Count vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Logisticka regresija sa Count vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'LR + Cv'))

#logisticka regresija sa count vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', CountVectorizer()),
    ('model', LogisticRegression(max_iter=1000, C=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Logisticka regresija sa Count vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Logisticka regresija sa Count vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'LR + Cv + L'))

#logisticka regresija sa TF-IDF vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', TfidfVectorizer()),
    ('model', LogisticRegression(max_iter=1000, C=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Logisticka regresija sa TF-IDF vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Logisticka regresija sa TF-IDF vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'LR + TF'))

#logisticka regresija sa TF-IDF vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', TfidfVectorizer()),
    ('model', LogisticRegression(max_iter=1000, C=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Logisticka regresija sa TF-IDF vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Logisticka regresija sa TF-IDF vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'LR + TF + L'))



#naivni bajesov klasifikator sa count vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', CountVectorizer()),
    ('model', MultinomialNB(alpha=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Naivni bajesov klasifikator sa Count vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Naivni bajesov klasifikator sa Count vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'NB + Cv'))

#Naivni bajesov klasifikator sa count vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', CountVectorizer()),
    ('model', MultinomialNB(alpha=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Naivni bajesov klasifikator sa Count vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Naivni bajesov klasifikator sa Count vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'NB + Cv + L'))

#Naivni bajesov klasifikator sa TF-IDF vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', TfidfVectorizer()),
    ('model', MultinomialNB(alpha=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Naivni bajesov klasifikator sa TF-IDF vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Naivni bajesov klasifikator sa TF-IDF vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'NB + TF'))

#Naivni bajesov klasifikator sa TF-IDF vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', TfidfVectorizer()),
    ('model', MultinomialNB(alpha=0.5))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Naivni bajesov klasifikator sa TF-IDF vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Naivni bajesov klasifikator sa TF-IDF vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'NB + TF + L'))



#stablo odlucivanja sa count vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', CountVectorizer()),
    ('model', DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Stablo odlucivanja sa Count vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Stablo odlucivanja sa Count vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'DT + Cv'))

#Stablo odlucivanja sa count vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', CountVectorizer()),
    ('model', DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'stablo odlucivanja sa Count vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Stablo odlucivanja sa Count vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'DT + Cv + L'))

#Stablo odlucivanja sa TF-IDF vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', TfidfVectorizer()),
    ('model', DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Stablo odlucivanja sa TF-IDF vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Stablo odlucivanja sa TF-IDF vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'DT + TF'))

#STablo odlucivanja sa TF-IDF vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', TfidfVectorizer()),
    ('model', DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Stablo odlucivanja sa TF-IDF vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Stablo odlucivanja sa TF-IDF vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'DT + TF + L'))



#Random Forest sa count vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', CountVectorizer()),
    ('model', RandomForestClassifier(n_estimators=150, max_depth=15, max_features='log2', min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Random Forest sa Count vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Random Forest sa Count vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'RF + Cv'))

#Random forest sa count vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', CountVectorizer()),
    ('model', RandomForestClassifier(n_estimators=150, max_depth=15, max_features='log2', min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Random forest sa Count vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Random Forest sa Count vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'RF + Cv + L'))

#Random Forest sa TF-IDF vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', TfidfVectorizer()),
    ('model', RandomForestClassifier(n_estimators=150, max_depth=15, max_features='log2', min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Random forest sa TF-IDF vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Random forest sa TF-IDF vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'RF + TF'))

#Random Forest sa TF-IDF vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', TfidfVectorizer()),
    ('model', RandomForestClassifier(n_estimators=150, max_depth=15, max_features='log2', min_samples_split=5, min_samples_leaf=2))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Random forest sa TF-IDF vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Random forest sa TF-IDF vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'RF + TF + L'))



#Extra trees sa count vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', CountVectorizer()),
    ('model', ExtraTreesClassifier(n_estimators=100, max_depth=10))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Extra trees sa Count vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Extra trees sa Count vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'ET + Cv'))

#extra trees sa count vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', CountVectorizer()),
    ('model', ExtraTreesClassifier(n_estimators=100, max_depth=10))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Extra trees sa Count vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Extra trees sa Count vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'ET + Cv + L'))

#Extra trees sa TF-IDF vektorizacijom bez lematizacije
pipeline = Pipeline([
    ('preprocessor', preprocessor_bez_lematizacije),
    ('vectorizer', TfidfVectorizer()),
    ('model', ExtraTreesClassifier(n_estimators=100, max_depth=10))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Extra trees sa TF-IDF vektorizacijom bez lematizacije: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Extra trees sa TF-IDF vektorizacijom bez lematizacije'
    best_pipeline = pipeline

rezultati.append((acc, 'ET + TF'))

#Extra trees sa TF-IDF vektorizacijom sa lematizacijom
pipeline = Pipeline([
    ('preprocessor', preprocessor_sa_lematizacijom),
    ('vectorizer', TfidfVectorizer()),
    ('model', ExtraTreesClassifier(n_estimators=100, max_depth=10))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Extra trees sa TF-IDF vektorizacijom sa lematizacijom: {acc:.4f}')
if acc > best_acc:
    best_acc = acc
    best_model = 'Extra trees sa TF-IDF vektorizacijom sa lematizacijom'
    best_pipeline = pipeline

rezultati.append((acc, 'ET + TF + L'))

print(f'\nNajvecu tacnost ima {best_model}, koja iznosi: {best_acc:.4f}')

vek = {
    ('Cv', CountVectorizer()),
    ('TF', TfidfVectorizer())
}
mod = {
    ('LR', LogisticRegression(max_iter=1000, C=0.5)),
    ('DT', DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2)),
    ('NB', MultinomialNB(alpha=0.5)),
    ('RF', RandomForestClassifier(n_estimators=150, max_depth=15, max_features='log2', min_samples_split=5, min_samples_leaf=2)),
    ('ET', ExtraTreesClassifier(n_estimators=100, max_depth=10))
}
pre = {
    ('+ L', preprocessor_sa_lematizacijom),
    ('- L', preprocessor_bez_lematizacije)
}
training_sizes = np.arange(0.55, 0.9, 0.05)
rezultati_velicina = []
for p in pre:
    for v in vek:
        for m in mod:
            rezultat = ([], p[0] + v[0] + m[0])
            for size in training_sizes:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=11)
                pipeline = Pipeline([
                    ('preprocessor', p[1]),
                    ('vectorizer', v[1]),
                    ('model', m[1])
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f'Veličina treninga: {size:.0%} | Tačnost: {acc:.4f}')
                rezultat[0].append((size, acc))
            rezultati_velicina.append(rezultat)

#testiranje uticaja broja tema samo na najboljoj kombinaciji modela i vektorizacije
prompts = df['prompt_name'].unique()
results = []
for k in range(1, len(prompts)):
    train_prompts = np.random.choice(prompts, size=k, replace=False)
    train_mask = df['prompt_name'].isin(train_prompts)
    X_train = df[train_mask]['text']
    y_train = df[train_mask]['label']
    X_test = df[~train_mask]['text']
    y_test = df[~train_mask]['label']
    if len(X_test) == 0:
        continue
    pipeline = best_pipeline
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((k, acc))


plt.figure()
plt.bar([rez[0] for rez in rezultati], [rez[1] for rez in rezultati], color='green')
plt.xlabel('Modeli')
plt.xticks(rotation=30)
plt.ylabel('Tacnost modela')
plt.title('Tacnost modela(velicina treninga 80%)')

for x in rezultati_velicina:
    plt.figure()
    plt.plot([rez[0] for rez in x[0]], [rez[1] for rez in x[0]], marker='o')
    plt.xlabel('Velicina treninga')
    plt.ylabel('Tacnost modela')
    plt.title(x[1])

plt.figure()
plt.plot([res[0] for res in results], [res[1] for res in results], marker='o')
plt.xlabel('Broj tema u treningu')
plt.ylabel('Tacnost modela')
plt.title('Uticaj broja tema na tacnost')
plt.grid(True)
plt.show()