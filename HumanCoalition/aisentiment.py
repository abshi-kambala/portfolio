import sqlite3
import pandas as pd
from transformers import pipeline
import pytesseract
from PIL import Image
import requests
from io import BytesIO


# connect to db and load posts
def load_posts_from_db(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT id, author, tweet_text, image_text FROM posts"
    posts_df = pd.read_sql(query, conn)
    conn.close()
    return posts_df


# load zero-shot classification models for classification
def load_classifiers():
    stance_classifier = pipeline("zero-shot-classification")
    category_classifier = pipeline("zero-shot-classification")

    return stance_classifier, category_classifier


# classify stance
def classify_stance(text, stance_classifier):
    try:
        if len(text.split()) < 5:  # dont classify if too short
            return "Unknown"

        candidate_labels = ["Pro-life", "Pro-choice"]
        result = stance_classifier(text, candidate_labels=candidate_labels)
        stance = result['labels'][0]  # label with the highest score

        # threshold
        if result['scores'][0] >= 0.4:
            return stance
        else:
            return "Unknown"  # classify as 'Unknown'
    except Exception as e:
        print(f"Error classifying stance: {e}")
        return "Unknown"


# classify secondary category(reasoning)
def classify_secondary_category(text, category_classifier):
    try:
        candidate_labels = [
            "Women's Rights", "Religion", "Health and Safety",
            "Personal Autonomy", "Social Justice", "Moral/Philosophical Beliefs", "Economic Impact"
        ]

        result = category_classifier(text, candidate_labels=candidate_labels)
        category = result['labels'][0]  # Get the label with the highest score

        # threshold for secondary categories
        if result['scores'][0] >= 0.4:
            return category
        else:
            return "Unknown"  # classify as 'Unknown' if threshold not met
    except Exception as e:
        print(f"Error classifying secondary category: {e}")
        return "Unknown"


# extract text from image URL (may not always work)
def extract_text_from_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        image_text = pytesseract.image_to_string(img)
        return image_text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None


# main processing function
def process_posts(posts_df, stance_classifier, category_classifier):
    # Classify stance (Pro-life, Pro-choice, Unknown) and secondary category for each post
    posts_df['stance'] = posts_df['tweet_text'].apply(lambda text: classify_stance(text, stance_classifier))
    posts_df['secondary_category'] = posts_df['tweet_text'].apply(
        lambda text: classify_secondary_category(text, category_classifier))

    # Extract image text if applicable
    posts_df['extracted_image_text'] = posts_df['image_text'].apply(
        lambda url: extract_text_from_image(url) if url else None)

    return posts_df


# save processed posts to new db
def save_classified_posts(posts_df, db_path):
    conn = sqlite3.connect(db_path)
    posts_df.to_sql('classified_posts', conn, if_exists='replace', index=False)
    conn.close()


# analysis of results
def analyze_results(posts_df):
    # Count the stances and secondary categories
    stance_counts = posts_df['stance'].value_counts()
    print("Stance distribution:\n", stance_counts)

    category_counts = posts_df['secondary_category'].value_counts()
    print("Secondary category distribution:\n", category_counts)


# main
if __name__ == "__main__":
    # Full path to your database (replace with your actual path)
    db_path = r'C:\Users\Abish\PycharmProjects\pythonProject\.venv\posts.db'  # Full path to your database
    posts_df = load_posts_from_db(db_path)

    # Load the models
    stance_classifier, category_classifier = load_classifiers()

    # Process the posts: classify stances, secondary categories, and extract image text
    processed_posts_df = process_posts(posts_df, stance_classifier, category_classifier)

    # Analyze the results (optional)
    analyze_results(processed_posts_df)

    # Save the processed posts to the new SQLite database (stances.db)
    new_db_path = r'C:\Users\Abish\PycharmProjects\pythonProject\.venv\stances.db'  # New database for saving the classified posts
    save_classified_posts(processed_posts_df, new_db_path)

    # Show a preview of the results
    print(processed_posts_df.head())
