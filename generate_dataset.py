

import csv
import random
import os

random.seed(42)

USERNAMES = [
    "sarah_j", "mike_23", "priya_k", "alex_99", "emma_w",
    "raj_s", "lisa_m", "john_d", "nina_p", "chris_b",
    "zara_h", "tom_r", "ana_g", "dev_k", "sofia_l",
    "ryan_m", "maya_s", "jake_t", "pooja_r", "luke_w",
    "diana_f", "sam_k", "tina_b", "vik_s", "bella_r",
    "harsh_p", "amy_c", "neil_m", "riya_d", "max_j"
]

POSITIVE_COMMENTS = [
    "This is absolutely amazing! Love it so much! ❤️",
    "Wow this made my day! You are so talented! 🔥",
    "Incredible work! Keep it up! You inspire me every day!",
    "This is the best thing I have seen all week! 😍",
    "So beautiful! I am obsessed with this content!",
    "You never disappoint! Always delivering quality content!",
    "This gave me goosebumps! Absolutely stunning work!",
    "I love everything about this! Pure perfection! ✨",
    "You are so gifted! This is truly inspiring!",
    "Omg this is everything! I cannot stop watching this!",
    "Blessed to see this on my feed today! Amazing!",
    "This deserves way more views! Incredible content!",
    "You just made my entire week better! Thank you!",
    "Absolutely love your content! Keep shining! 🌟",
    "This is so wholesome and beautiful! Heart is full!",
    "Best creator on this platform hands down! Love you!",
    "This video is pure gold! Sharing with everyone!",
    "You are incredible! Never stop creating please!",
    "This made me smile so much! Thank you for this!",
    "Genuinely the most talented person I follow! 🙌",
]

NEGATIVE_COMMENTS = [
    "This is the worst content I have ever seen! 😡",
    "Absolutely terrible! Unfollow! Waste of my time!",
    "You have no talent whatsoever! Stop posting please!",
    "This is so bad I cannot even watch it fully!",
    "Disgusting content! How does this even get views?",
    "You are such a fake person! Nobody likes this!",
    "This made me angry! Complete waste of time!",
    "Horrible! Just horrible! Delete this immediately!",
    "I hate everything about this! So disappointing!",
    "You ruined my mood with this terrible content! 😤",
    "This is trash! Absolute garbage content!",
    "Stop embarrassing yourself by posting this stuff!",
    "Worst video on the internet! Disliked and reported!",
    "You have zero creativity! This is so boring and bad!",
    "I regret watching this! Two minutes of my life wasted!",
    "This is so cringe! I cannot believe people watch this!",
    "Pathetic attempt at content creation! Just stop!",
    "You clearly have no idea what you are doing! Terrible!",
    "This gave me a headache! So annoying and pointless!",
    "Garbage content from a garbage creator! Unsubscribed!",
]

NEUTRAL_COMMENTS = [
    "Okay I guess. Nothing special about this.",
    "Watched it. Moving on with my day.",
    "It is what it is. Neither good nor bad.",
    "Average content. Has been done before many times.",
    "Not my style but I can see why others like it.",
    "Interesting approach. Not sure how I feel about it.",
    "This exists I suppose. Good for you.",
    "Seen better seen worse. Just average content.",
    "It is fine I guess. Nothing memorable though.",
    "Okay video. Forgot about it already.",
]

TOXIC_COMMENTS = [
    "You are such an idiot! Nobody wants to see your face!",
    "Kill this channel! You are absolutely worthless!",
    "Stupid moron! Get off the internet forever!",
    "You are ugly and talentless! Go away permanently!",
    "Everyone hates you! Just quit already you loser!",
    "What a complete waste of human existence you are!",
    "You are so dumb it hurts to watch! Pathetic loser!",
    "Nobody likes you! You are a complete embarrassment!",
    "Get lost you absolute idiot! Worst creator ever!",
    "You make me sick! Disgusting excuse for a creator!",
]

EXCITED_COMMENTS = [
    "OMG OMG OMG I cannot believe this!!! 🎉🎉🎉",
    "YESSS THIS IS EVERYTHING I NEEDED TODAY!!!",
    "I AM SCREAMING RIGHT NOW THIS IS SO GOOD!!!",
    "Wait WHAT?! This is insane!! I am obsessed!!!",
    "NO WAY!! This is literally the best thing ever!!",
    "I CANNOT HANDLE THIS!! TOO GOOD TOO GOOD!! 🔥🔥",
    "THIS JUST MADE MY ENTIRE YEAR!! AMAZING!!",
    "WAIT WAIT WAIT this is unbelievably good!!!",
]

SAD_COMMENTS = [
    "This made me cry so much 😢 So beautiful and sad",
    "Watching this after a breakup and I am in tears 💔",
    "This hit different today. Going through hard times 😭",
    "I needed this today. Life has been so tough lately",
    "Crying watching this. Miss my old life so much 😢",
    "This broke my heart in the best possible way 💔",
    "So emotional watching this. Thank you for sharing",
    "This brought back so many painful memories 😢",
]

QUESTIONS = [
    "What camera do you use for this? Looks amazing!",
    "Where was this filmed? Looks so beautiful!",
    "Can you do a tutorial on how you made this?",
    "What song is playing in the background?",
    "How long did this take to make?",
    "What editing software do you use?",
    "Is this available to buy somewhere?",
    "Can you share the recipe or process for this?",
]

def generate_dataset(n=200, path="instagram_comments.csv"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    rows = []
    post_ids = [f"POST{i:03d}" for i in range(1, 21)]
    
    # Distribution of comments
    distributions = [
        (POSITIVE_COMMENTS, 50, "positive", "happy", False),
        (NEGATIVE_COMMENTS, 40, "negative", "angry", False),
        (NEUTRAL_COMMENTS,  20, "neutral",  "neutral", False),
        (TOXIC_COMMENTS,    30, "negative", "angry", True),
        (EXCITED_COMMENTS,  20, "positive", "excited", False),
        (SAD_COMMENTS,      20, "negative", "sad", False),
        (QUESTIONS,         20, "neutral",  "neutral", False),
    ]
    
    for comment_list, count, sentiment, emotion, is_toxic in distributions:
        for _ in range(count):
            comment = random.choice(comment_list)
            rows.append({
                "comment_id": None,
                "username": random.choice(USERNAMES),
                "post_id": random.choice(post_ids),
                "comment_text": comment,
                "sentiment": sentiment,
                "emotion": emotion,
                "is_toxic": int(is_toxic),
                "likes": random.randint(0, 500),
            })
    
    # Shuffle and assign IDs
    random.shuffle(rows)
    rows = rows[:n]
    for i, row in enumerate(rows, 1):
        row["comment_id"] = f"CMT{i:04d}"
    
    fields = ["comment_id", "username", "post_id", "comment_text",
              "sentiment", "emotion", "is_toxic", "likes"]
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    
    toxic = sum(1 for r in rows if r["is_toxic"] == 1)
    pos   = sum(1 for r in rows if r["sentiment"] == "positive")
    neg   = sum(1 for r in rows if r["sentiment"] == "negative")
    neu   = sum(1 for r in rows if r["sentiment"] == "neutral")
    
    print(f"✅ Dataset generated: {path}")
    print(f"   Total: {len(rows)}")
    print(f"   Positive: {pos} | Negative: {neg} | Neutral: {neu}")
    print(f"   Toxic: {toxic} | Non-Toxic: {len(rows)-toxic}")
    return path

if __name__ == "__main__":
    generate_dataset(200, "instagram_comments.csv")






