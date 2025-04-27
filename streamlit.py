import streamlit as st
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

def simple_tokenizer(text):
    return text.lower().split()

@st.cache_data
def load_data():
    data = pd.read_csv('combined_output.csv')
    data['metadata'] = data['Genre'] + ' ' + data['Cast'] + ' ' + data['Synopsis']
    return data

data = load_data()

@st.cache_resource
def train_doc2vec(data):
    documents = [
        TaggedDocument(words=simple_tokenizer(text), tags=[str(i)])
        for i, text in enumerate(data['metadata'])
    ]
    model = Doc2Vec(vector_size=150, window=5, min_count=2, workers=4, epochs=40, negative=5)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    
    doc_vectors = [model.infer_vector(simple_tokenizer(text)) for text in data['metadata']]
    return np.array(doc_vectors)

doc_vectors = train_doc2vec(data)

@st.cache_resource
def setup_collab(data):
    features = data[['Rating', 'Number of Votes']].copy()
    features.fillna(features.mean(), inplace=True)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(features_scaled)
    return knn, features_scaled

knn, collab_scaled = setup_collab(data)

def recommend_hybrid(title, top_n=5):
    idx = data[data['Title'] == title].index[0]
    
    query_vector = doc_vectors[idx].reshape(1, -1)
    content_sim = cosine_similarity(query_vector, doc_vectors)[0]
    
    _, collab_indices = knn.kneighbors([collab_scaled[idx]], n_neighbors=top_n + 1)
    collab_indices = collab_indices[0][1:]
    
    hybrid_scores = content_sim.copy()
    hybrid_scores[collab_indices] += 1
    
    top_indices = hybrid_scores.argsort()[::-1][1:top_n+1]
    return data['Title'].iloc[top_indices]

st.title('🎬 TV Series Recommender System (Hybrid Model)')

selected_title = st.selectbox('Select a TV Show:', data['Title'].values)

if st.button('Get Recommendations'):
    recommendations = recommend_hybrid(selected_title)
    st.subheader('Recommended TV Shows:')
    for show in recommendations:
        st.write(show)
