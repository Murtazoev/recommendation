import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def simple_tokenizer(text):
    return str(text).lower().split()

@st.cache_data
def load_data():
    data = pd.read_csv('combined_output.csv')
    data = data.drop_duplicates(subset = 'Title')
    data['metadata'] = data['Genre'] + ' ' + data['Cast'] + ' ' + data['Synopsis']
    sample_data = data.sample(n=10000, random_state=42)
    return sample_data.reset_index(drop=True)

data = load_data()

@st.cache_resource
def load_doc2vec_model():
    model = Doc2Vec.load('model.d2v')
    return model

model = load_doc2vec_model()

@st.cache_resource
def generate_doc_vectors(data, _model):
    doc_vectors = [_model.infer_vector(simple_tokenizer(text)) for text in data['metadata']]
    return np.array(doc_vectors)

doc_vectors = generate_doc_vectors(data, model)

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

def recommend_hybrid(selected_titles, top_n=5):
    if not selected_titles:
        return []

    indices = data[data['Title'].isin(selected_titles)].index.tolist()

    query_vectors = np.array([doc_vectors[idx] for idx in indices])
    avg_query_vector = np.mean(query_vectors, axis=0).reshape(1, -1)

    content_sim = cosine_similarity(avg_query_vector, doc_vectors)[0]

    _, collab_indices = knn.kneighbors([np.mean(collab_scaled[indices], axis=0)], n_neighbors=top_n + len(selected_titles) + 1)
    collab_indices = collab_indices[0]

    hybrid_scores = content_sim.copy()
    hybrid_scores[collab_indices] += 1

    for idx in indices:
        hybrid_scores[idx] = -np.inf

    top_indices = hybrid_scores.argsort()[::-1][:top_n]
    return data['Title'].iloc[top_indices]


st.title('🎬 TV Series Recommender System (Hybrid Model)')

selected_title = st.selectbox('Select a TV Show:', data['Title'].values)
selected_titles = st.multiselect('Select one or more TV Shows:', sample_data['Title'].values)

top_k = st.slider('How many recommendations?', 1, 20, 5)

if st.button('Get Recommendations'):
    recommendations = recommend_hybrid(selected_title, top_n=top_k)
    st.subheader('Recommended TV Shows:')
    for show in recommendations:
        st.write(show)
