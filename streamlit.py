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

def recommend_hybrid(title, top_n=5):
    idx = data[data['Title'] == title].index[0]

    query_vector = doc_vectors[idx].reshape(1, -1)
    content_sim = cosine_similarity(query_vector, doc_vectors)[0]

    _, collab_indices = knn.kneighbors([collab_scaled[idx]], n_neighbors=top_n + 1)
    collab_indices = collab_indices[0][1:]

    hybrid_scores = content_sim.copy()
    hybrid_scores[collab_indices] += 1

    hybrid_scores[idx] = -np.inf

    top_indices = hybrid_scores.argsort()[::-1][1:top_n+1]
    return data['Title'].iloc[top_indices]

# --- Streamlit App ---
st.title('🎬 TV Series Recommender System (Hybrid Model)')

selected_title = st.selectbox('Select a TV Show:', data['Title'].values)

top_k = st.slider('How many recommendations?', 1, 20, 5)

if st.button('Get Recommendations'):
    recommendations = recommend_hybrid(selected_title, top_n=top_k)
    st.subheader('Recommended TV Shows:')
    for show in recommendations:
        st.write(show)
