import sqlite3
import pandas as pd


# classified posts database
def load_classified_posts(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT id, author, tweet_text, stance, secondary_category FROM classified_posts"
    posts_df = pd.read_sql(query, conn)
    conn.close()
    return posts_df


# by label
def count_labels(posts_df):

    stance_counts = posts_df['stance'].value_counts()
    print("Stance label counts:\n", stance_counts)


    category_counts = posts_df['secondary_category'].value_counts()
    print("\nSecondary category label counts:\n", category_counts)


# count tweets with specific combinations of stance and secondary category
def count_combined_labels(posts_df, stance_label, category_label):
    # organize posts that match both the given stance and secondary category
    combined_count = len(posts_df[(posts_df['stance'] == stance_label) &
                                  (posts_df['secondary_category'] == category_label)])
    print(f"\nNumber of tweets labeled as {stance_label} and {category_label}: {combined_count}")
    return combined_count


# sort the database by stance or secondary category
def sort_by_label(posts_df, label="stance"):
    sorted_df = posts_df.sort_values(by=label)
    return sorted_df


# main
if __name__ == "__main__":

    db_path = r'C:\Users\Abish\PycharmProjects\pythonProject\.venv\stances.db'  # Full path to your classified posts database

    # Load the classified posts
    posts_df = load_classified_posts(db_path)

    # count labels
    count_labels(posts_df)

    # count combinations of stance and secondary category
    # prolife combinations
    prolife_combination_counts = [
        ("Pro-life", "Religion"),
        ("Pro-life", "Women's Rights"),
        ("Pro-life", "Personal Autonomy"),
        ("Pro-life", "Health and Safety"),
        ("Pro-life", "Social Justice"),
        ("Pro-life", "Moral/Philosophical Beliefs"),
        ("Pro-life", "Economic Impact")
    ]

    for stance, category in prolife_combination_counts:
        count_combined_labels(posts_df, stance, category)

    # prochoice combinations
    prochoice_combination_counts = [
        ("Pro-choice", "Religion"),
        ("Pro-choice", "Women's Rights"),
        ("Pro-choice", "Personal Autonomy"),
        ("Pro-choice", "Health and Safety"),
        ("Pro-choice", "Social Justice"),
        ("Pro-choice", "Moral/Philosophical Beliefs"),
        ("Pro-choice", "Economic Impact")
    ]

    for stance, category in prochoice_combination_counts:
        count_combined_labels(posts_df, stance, category)

    # sort the posts by stance
    sorted_posts = sort_by_label(posts_df, label="stance")

    # Display the sorted database
    print("\nSorted posts by stance (top 10):\n", sorted_posts.head(10))
