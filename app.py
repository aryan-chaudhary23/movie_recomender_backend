from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import numpy as np

# Load data and model with correct relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get backend folder path
df = pd.read_csv(os.path.join(BASE_DIR, "newmovie_df.csv"))

# Load similarity matrix from .npz
simil_data = np.load(os.path.join(BASE_DIR, "similarity.npz"))
simil = simil_data["similarity"]  # Extract the matrix

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "../frontend/build"), template_folder=os.path.join(BASE_DIR, "../frontend/build"))

@app.route("/api/movies", methods=["GET"])
def all_movies():
    present_movies = df["title"].unique().tolist()
    return jsonify({"movies_list": present_movies})

@app.route("/api/recomend", methods=["POST"])
def recommend_movies():
    data = request.get_json()
    movie_name = data.get("movie")
    movie_list, id_list = recommend(movie_name)
    return jsonify({"movie_list": movie_list, "id_list": id_list})

def recommend(movie):
    """Returns top 5 recommended movies and their TMDB IDs."""
    movie_index = df[df['title'] == movie].index[0]
    distances = simil[movie_index]  # Use the similarity matrix loaded from .npz
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    tmdb_ids = []
    for i in movies_list:
        recommended_movies.append(df.iloc[i[0]].title)
        tmdb_ids.append(int(df.iloc[i[0]].tmdbId))
    return recommended_movies, tmdb_ids

# Serve React Frontend
@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# Allow CORS
CORS(app, origins=["https://movie-recomender-frontend.onrender.com"])

if __name__ == "__main__":
    from waitress import serve
    app.run(debug=True)
