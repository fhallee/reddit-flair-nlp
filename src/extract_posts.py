import praw
from prawcore.exceptions import ResponseException
from datetime import datetime
import sqlite3

def main():
    # Get Reddit API credentials from the user
    client_id = input("Enter your Reddit client ID: ")
    client_secret = input("Enter your Reddit client secret: ")
    user_agent = input("Enter your Reddit user agent: ")

    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    # Define the subreddit and flairs of interest
    subreddit = reddit.subreddit("OCD")
    flairs_of_interest = [
        "Venting",
        "Achievement",
        "Question about OCD and mental illness",
        "I need support - advice welcome",
        "I just need to vent",
        "Crisis",
        "Sharing a Win!",
    ]

    # Connect to SQLite database and create posts table
    conn = sqlite3.connect("../data/posts.db")
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS posts''')
    c.execute('''CREATE TABLE posts (
            created_utc TEXT,
            text TEXT,
            flair TEXT
    )''')

    # Define function to insert posts into the database
    def insert_post_data(created_utc, text, flair):
        c.execute("INSERT INTO posts VALUES (?, ?, ?)",
                (created_utc, text, flair))
        conn.commit()

    try:
        # Fetch and insert posts for each flair
        for flair in flairs_of_interest:
            query = f"flair:{flair}"
            submissions = subreddit.search(query=query, limit=500)
            counter = 0
            for submission in submissions:
                created_utc = datetime.fromtimestamp(submission.created_utc)
                text = submission.title + submission.selftext
                flair = submission.link_flair_text
                # Transform flair to "Non_crisis" if it's not "Crisis"
                if flair != "Crisis":
                    transformed_flair = "Non_crisis"
                else:
                    transformed_flair = flair
                insert_post_data(created_utc, text, transformed_flair)
                counter = counter + 1
    except ResponseException:
        # Handle invalid credentials error
        print("Error occured while fetching submissions. Check that your credentials were entered properly.")
        return

    # Print the count of posts for each flair
    print("Data extraction complete.")
    c.execute("SELECT flair, COUNT(*) FROM posts GROUP BY flair")
    results = c.fetchall()
    for row in results:
            print(f"{row[0]}: {row[1]} posts")

    conn.close()

if __name__ == "__main__":
    main()
