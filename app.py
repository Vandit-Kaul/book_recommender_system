from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv", dtype=str, low_memory=False)  # Fix dtype warning

# Merge Ratings with Books
ratings_with_name = ratings.merge(books, on="ISBN")

# Calculate Number of Ratings Per Book
num_rating_df = ratings_with_name.groupby("Book-Title").count()["Book-Rating"].reset_index()
num_rating_df.rename(columns={"Book-Rating": "num_ratings"}, inplace=True)

# Calculate Average Rating Per Book
avg_rating_df = ratings_with_name.groupby("Book-Title", as_index=False)["Book-Rating"].mean()
avg_rating_df.rename(columns={"Book-Rating": "avg_rating"}, inplace=True)

# Merge Popular Books Data
popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
popular_df = popular_df[popular_df["num_ratings"] >= 250].sort_values("avg_rating", ascending=False).head(50)

# Add Author and Image
popular_df = popular_df.merge(books, on="Book-Title", how="left") \
    .drop_duplicates(subset=["Book-Title"])[["Book-Title", "Book-Author", "Image-URL-M", "num_ratings", "avg_rating"]]

# Save Data for Future Use
pickle.dump(popular_df, open("popular.pkl", "wb"))

# Collaborative Filtering
x = ratings_with_name.groupby("User-ID").count()["Book-Rating"] > 200
filtered_users = x[x].index
filtered_rating = ratings_with_name[ratings_with_name["User-ID"].isin(filtered_users)]

y = filtered_rating.groupby("Book-Title").count()["Book-Rating"] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating["Book-Title"].isin(famous_books)]
pt = final_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
pt.fillna(0, inplace=True)

# Similarity Scores
similarity_scores = cosine_similarity(pt)


def get_recommendations(book_name):
    if book_name not in pt.index:
        return []  # Return empty list instead of error message

    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    recommended_books = []
    for i in similar_items:
        book_title = pt.index[i[0]]
        book_data = books[books["Book-Title"] == book_title].drop_duplicates("Book-Title")
        recommended_books.append({
            "title": book_title,
            "author": book_data["Book-Author"].values[0] if not book_data.empty else "Unknown",
            "image": book_data["Image-URL-M"].values[0] if not book_data.empty else "#"
        })

    return recommended_books


# Flask App
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html",
                           book_name=popular_df["Book-Title"].values,
                           author=popular_df["Book-Author"].values,
                           image=popular_df["Image-URL-M"].values,
                           votes=popular_df["num_ratings"].values,
                           rating=popular_df["avg_rating"].values)


@app.route('/recommend_books', methods=['POST', 'GET'])
def recommend_books():
    recommendations = []
    if request.method == "POST":
        book_name = request.form.get("book")
        if book_name:
            recommendations = get_recommendations(book_name)

    return render_template("recommendations.html", recommendations=recommendations)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
