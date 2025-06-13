import sqlite3
import pytesseract
from PIL import Image
import requests
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import random
import Levenshtein


options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--headless")
options.add_argument("--disable-gpu")  # if headless
options.add_argument("--no-sandbox")
options.add_argument("window-size=1920x1080")  # window size
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# connect to SQLite database
def create_db():
    conn = sqlite3.connect('posts.db')
    cursor = conn.cursor()

    # Create table only if it does not exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        author TEXT,
        tweet_text TEXT,
        image_text TEXT,
        UNIQUE (author, tweet_text)
    )''')

    conn.commit()
    conn.close()

# extract text from an image using OCR
def extract_text_from_image(image_url, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            # get the image
            response = requests.get(image_url)
            if response.status_code != 200:
                print(f"Failed to fetch image, status code: {response.status_code}")
                return None

            # open image
            img = Image.open(BytesIO(response.content))

            # convert to a compatible format (RGB)
            img = img.convert('RGB')

            # OCR using pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            text = pytesseract.image_to_string(img)

            if text.strip() == "":
                print("No text found in the image.")
                return None

            return text

        except Exception as e:
            attempt += 1
            print(f"Error extracting text from image on attempt {attempt}: {e}")
            if attempt >= retries:
                return None

# normalize tweet text
def normalize_text(text):
    # Remove extra whitespace, newlines, spaces
    normalized = re.sub(r'\s+', ' ', text.strip())
    normalized = re.sub(r'[^\w\s.,!?;:]', '', normalized)
    normalized = normalized.lower()  # lower case everything
    return normalized

# log into Twitter function
def login_to_twitter(username, password):
    try:
        driver.get("https://twitter.com/login")
        time.sleep(2)

        # username
        username_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//input[@name="text"]'))
        )
        username_input.send_keys(username)
        username_input.send_keys(Keys.RETURN)

        # wait so twitter has time to load
        time.sleep(2)

        # password
        password_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//input[@name="password"]'))
        )
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)

        # wait for login to complete
        time.sleep(5)

        print("Login successful!")

    except Exception as e:
        print(f"Login failed: {e}")
        driver.quit()

# check if a tweet is already in the database
def is_duplicate_in_db(author, tweet_text, threshold=0.85):
    try:
        normalized_tweet_text = normalize_text(tweet_text)
        normalized_author_handle = normalize_text(author)

        print(f"Checking for duplicates: Author: {normalized_author_handle}, Tweet: {normalized_tweet_text}")  # Add more debug info

        conn = sqlite3.connect('posts.db')
        cursor = conn.cursor()

        cursor.execute('''SELECT tweet_text FROM posts WHERE author = ?''', (normalized_author_handle,))
        previous_tweets = cursor.fetchall()

        for prev_tweet in previous_tweets:
            prev_tweet_text = prev_tweet[0]
            similarity = Levenshtein.ratio(normalized_tweet_text, prev_tweet_text)

            # duplicate check
            if similarity > threshold:
                print(f"Duplicate detected (similarity {similarity}): {normalized_tweet_text}")
                return True

        return False

    except Exception as e:
        print(f"Error checking database for duplicates: {e}")
        return False
    finally:
        conn.close()

# get the current number of saved posts in the database
def get_saved_post_count():
    try:
        conn = sqlite3.connect('posts.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM posts")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error fetching saved posts count: {e}")
        return 0

# scrape tweets by hashtag and save a specified number of posts to the database
def scrape_tweets_by_hashtag(hashtag, posts_to_save):
    try:
        driver.get(f"https://twitter.com/hashtag/{hashtag}?src=hashtag_click")
        time.sleep(3)

        posts = []
        seen_posts = set()
        saved_post_count = get_saved_post_count()  # Get the current number of saved posts

        while saved_post_count < posts_to_save:
            tweet_elements = driver.find_elements(By.XPATH, '//div[@aria-label="Timeline: Search timeline"]//article')
            current_tweet_count = len(tweet_elements)

            # Debug: Print number of tweets loaded
            print(f"Found {current_tweet_count} tweet elements.")

            # Extract tweet info if new tweets are found
            for tweet in tweet_elements:
                try:
                    tweet_text = tweet.find_element(By.XPATH, './/div[@dir="auto"]/span').text
                    author_handle = tweet.find_element(By.XPATH, './/div[@dir="ltr"]/span').text

                    if (author_handle, tweet_text) in seen_posts:
                        continue  # Skip already seen posts

                    # Check for duplicates in the database before saving
                    if is_duplicate_in_db(author_handle, tweet_text):
                        print(f"Duplicate tweet found: {author_handle} - {tweet_text}")
                        continue

                    image_urls = [img.get_attribute('src') for img in tweet.find_elements(By.XPATH, './/img[@src]')]
                    image_text = ""
                    for url in image_urls:
                        extracted_text = extract_text_from_image(url)
                        if extracted_text:
                            image_text += extracted_text + "\n"

                    # Add post to list
                    posts.append({"author": author_handle, "tweet_text": tweet_text, "image_text": image_text.strip()})
                    seen_posts.add((author_handle, tweet_text))

                    # Save post to the database if it's not a duplicate
                    save_posts_to_db([{"author": author_handle, "tweet_text": tweet_text, "image_text": image_text.strip()}])
                    saved_post_count += 1

                    # Break if the target saved posts count is reached
                    if saved_post_count >= posts_to_save:
                        break

                except Exception as e:
                    print(f"Error extracting tweet: {e}")

            # Scroll down and wait for new tweets to load
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(random.uniform(6, 10))  # Increase time to allow loading

            # Debug: Check if the tweet count is increasing
            if current_tweet_count == len(tweet_elements):
                print("No new tweets loaded. Reached the bottom of the page.")
                break

        print(f"Saved {saved_post_count} posts to the database.")

    except Exception as e:
        print(f"Error scraping tweets: {e}")

# save posts to SQLite database
def save_posts_to_db(posts):
    try:
        # Connect to the database
        conn = sqlite3.connect('posts.db')  # Change to posts.db
        cursor = conn.cursor()

        # Insert posts into the database, but check for duplicates first
        for post in posts:
            normalized_tweet_text = normalize_text(post['tweet_text'])
            normalized_author_handle = normalize_text(post['author'])

            try:
                cursor.execute('''INSERT INTO posts (author, tweet_text, image_text) 
                                 VALUES (?, ?, ?)''',
                               (normalized_author_handle, normalized_tweet_text, post['image_text'] if post['image_text'] else None))
            except sqlite3.IntegrityError:
                # Duplicate post detected, skip insertion
                print(f"Duplicate post found: Author: {post['author']} Post: {post['tweet_text']}")
                continue

        conn.commit()
        print(f"Saved {len(posts)} posts to the database.")
    except Exception as e:
        print(f"Error saving posts to database: {e}")
    finally:
        conn.close()

# main
def main():
    username = "BruceWayne4747"  # Replace with your Twitter username
    password = "LukaDoncic@77"  # Replace with your Twitter password
    hashtag = "inducedabortion"  # Hashtag you want to scrape
    num_tweets = 1300  # Target number of unique posts to save

    # database
    create_db()

    # login
    login_to_twitter(username, password)

    # scraping
    scrape_tweets_by_hashtag(hashtag, num_tweets)


    driver.quit()

if __name__ == "__main__":
    main()
