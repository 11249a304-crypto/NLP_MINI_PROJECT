"""
generate_dataset.py
-------------------
Builds data/feedback_dataset.csv  (columns: id, feedback, aspect, sentiment).

The dataset has two parts:
  1. 60 hand-written, realistic student comments (4 per aspect per sentiment).
  2. Several hundred comments generated from sentence templates, with a little
     random noise (typos, lower-casing, filler words, negations like "not helpful")
     so the data is not perfectly clean.

NOTE: This is a *synthetic* dataset made for demonstration. For a stronger
project, collect 50-100 REAL comments (for example with a Google Form), label
them, and append them to the CSV - the rest of the code works unchanged.

Run:  python generate_dataset.py
"""
import csv
import random
from pathlib import Path

random.seed(42)

OUT_FILE = Path(__file__).parent / "data" / "feedback_dataset.csv"

# --------------------------------------------------------------------------
# Template vocabulary, per aspect
# --------------------------------------------------------------------------
ASPECTS = {
    "Teaching": {
        "nouns": [("faculty", "are"), ("lectures", "are"), ("teaching methods", "are"),
                  ("professors", "are"), ("teaching quality", "is"), ("teachers", "are"),
                  ("classroom sessions", "are"), ("lecturers", "are"),
                  ("staff", "are"), ("assignments", "are")],
        "pos": ["excellent", "outstanding", "very supportive", "highly knowledgeable", "engaging",
                "inspiring", "very clear", "impressive", "helpful", "motivating"],
        "neg": ["boring", "poor", "unhelpful", "disappointing", "uninspiring", "confusing",
                "below average", "very weak", "terrible"],
        "negpos": ["helpful", "clear", "engaging", "satisfactory", "good", "interesting"],
    },
    "Course Content": {
        "nouns": [("syllabus", "is"), ("course content", "is"), ("subjects", "are"),
                  ("study materials", "are"), ("assignments", "are"), ("practical sessions", "are"),
                  ("curriculum", "is"), ("projects", "are"), ("course structure", "is"),
                  ("elective options", "are"), ("lab sessions", "are"), ("labs", "are")],
        "pos": ["well structured", "relevant", "up to date", "interesting", "very useful",
                "practical", "excellent", "comprehensive", "valuable", "industry oriented"],
        "neg": ["outdated", "too theoretical", "irrelevant", "overloaded", "boring",
                "poorly organized", "too heavy", "confusing", "repetitive", "lacking depth"],
        "negpos": ["relevant", "useful", "interesting", "practical", "helpful", "up to date"],
    },
    "Exams": {
        "nouns": [("exams", "are"), ("question papers", "are"), ("evaluation process", "is"),
                  ("internal assessments", "are"), ("grading", "is"), ("exam schedule", "is"),
                  ("practical exams", "are"), ("marking scheme", "is"),
                  ("timetable", "is"), ("assignments", "are")],
        "pos": ["fair", "well balanced", "transparent", "reasonable", "well organized", "clear",
                "satisfactory", "well planned"],
        "neg": ["too difficult", "unfair", "unreasonable", "badly scheduled", "too lengthy",
                "confusing", "stressful", "poorly planned", "too tough"],
        "negpos": ["fair", "reasonable", "balanced", "clear", "transparent"],
    },
    "Facilities": {
        "nouns": [("library", "is"), ("laboratories", "are"), ("classrooms", "are"),
                  ("canteen", "is"), ("hostel facilities", "are"), ("sports facilities", "are"),
                  ("computer labs", "are"), ("restrooms", "are"), ("campus infrastructure", "is"),
                  ("drinking water facilities", "are"), ("labs", "are"), ("lab equipment", "is")],
        "pos": ["excellent", "well maintained", "spacious", "clean", "modern", "well equipped",
                "very good", "comfortable", "great"],
        "neg": ["poorly maintained", "overcrowded", "dirty", "outdated", "very bad", "inadequate",
                "uncomfortable", "terrible", "too small", "unhygienic"],
        "negpos": ["clean", "spacious", "well maintained", "comfortable", "adequate", "well equipped"],
    },
    "Administration": {
        "nouns": [("administration", "is"), ("office staff", "are"), ("fee structure", "is"),
                  ("admission process", "is"), ("student support services", "are"),
                  ("timetable", "is"), ("placement cell", "is"), ("scholarship process", "is"),
                  ("grievance system", "is"), ("staff", "are")],
        "pos": ["very helpful", "efficient", "transparent", "responsive", "well organized",
                "prompt", "cooperative", "smooth", "friendly"],
        "neg": ["slow", "unresponsive", "disorganized", "bureaucratic", "unhelpful",
                "poorly managed", "confusing", "rude", "inefficient", "unfriendly"],
        "negpos": ["helpful", "responsive", "efficient", "cooperative", "organized"],
    },
}

NEUTRAL_ADJ = ["average", "okay", "acceptable", "decent", "fine", "ordinary", "just okay",
               "neither good nor bad", "moderate"]

POS_TAILS = ["and I am quite happy with it", "which makes a real difference",
             "and I would recommend it to juniors", "and it has helped me a lot",
             "which keeps students motivated", "and I hope it stays this way",
             "so I never had any problem", "and my friends agree", "which I truly appreciate",
             "and it is better than my previous college", "and it makes student life easier",
             "so I am grateful for it"]
NEG_TAILS = ["and it needs urgent improvement", "which affects students badly",
             "and I am disappointed", "and many of my classmates feel the same",
             "and nothing has changed for years", "which is really frustrating",
             "and we have complained many times", "so it wastes our time",
             "and I hope the management looks into it", "which is not acceptable",
             "and it was better in my school", "so students are unhappy"]
NEU_TAILS = ["but nothing special", "and I have no strong opinion",
             "and there is some room to improve", "it does the job",
             "so I have nothing more to add", "and it is the same as last year"]

# Adjectives that already carry an intensifier / sound odd with one
NO_ADVERB_FIRST = {"very", "too", "highly", "not", "extremely", "neither", "just", "below",
                   "excellent", "outstanding", "terrible", "impressive", "great", "well",
                   "badly", "poorly", "lacking", "up"}

FRAMES = [
    "The {n} {be} {adj}.",
    "Our {n} {be} {adj}.",
    "I feel the {n} {be} {adj}.",
    "In my opinion, the {n} {be} {adj}.",
    "I think the {n} {be} {adj}.",
    "Honestly, the {n} {be} {adj}, {tail}.",
    "To be frank, the {n} {be} {adj}, {tail}.",
]
NEGATION_FRAMES = [
    "The {n} {be} not {adj}.",
    "I feel the {n} {be} not {adj} at all.",
    "Sadly, the {n} {be} not {adj}.",
]

# --------------------------------------------------------------------------
# Hand-written comments (aspect -> sentiment -> list)
# --------------------------------------------------------------------------
HAND = {
    "Teaching": {
        "Positive": [
            "Our professors explain difficult topics in a very simple way and always clear our doubts.",
            "I really enjoy the lectures because the staff use real life examples to teach.",
            "My mentor is very supportive and guides us even after class hours.",
            "The teachers are passionate about the subject and make every class interesting.",
        ],
        "Negative": [
            "Most lectures are just reading out the slides, I learn nothing in class.",
            "Some faculty members never come prepared and the classes are very boring.",
            "The teacher goes too fast and does not bother whether students understood.",
            "Doubts are ignored in class and the staff are not approachable at all.",
        ],
        "Neutral": [
            "Teaching is fine, nothing exceptional but nothing to complain about.",
            "Some lectures are good while others are average.",
            "Classes are held five days a week and each lecture lasts fifty minutes.",
            "The teaching style is traditional and it works okay for me.",
        ],
    },
    "Course Content": {
        "Positive": [
            "The syllabus covers the latest industry topics and the projects are really useful.",
            "I like that the course has a good balance of theory and practical work.",
            "Study materials shared by the department are well organised and easy to follow.",
            "The elective subjects give us a lot of choice and are relevant to jobs.",
        ],
        "Negative": [
            "The syllabus is outdated and has nothing about current technologies.",
            "Too many subjects in one semester and the content is overloaded.",
            "Most topics are theoretical and we never see how they are used in practice.",
            "Assignments are repetitive and do not help us understand the subject.",
        ],
        "Neutral": [
            "The course has five units and each unit is covered in about three weeks.",
            "The syllabus is standard and similar to other universities.",
            "Some subjects are interesting, some are just okay.",
            "Study materials are provided through the online portal.",
        ],
    },
    "Exams": {
        "Positive": [
            "Question papers are fair and cover what was taught in class.",
            "The internal assessments help us to prepare well for the end semester exam.",
            "Results are published on time and the evaluation is transparent.",
            "I like the continuous assessment system because it reduces exam stress.",
        ],
        "Negative": [
            "The exam papers were very lengthy and we did not get enough time to finish.",
            "Valuation is unfair, marks are not given even for correct answers.",
            "Exam timetable is too tight with no gap between papers.",
            "Questions come from outside the syllabus which is very unfair to students.",
        ],
        "Neutral": [
            "The end semester exams are conducted in December and May.",
            "Internal exams carry forty percent of the total marks.",
            "The exam pattern is the same as last year.",
            "Exams were of moderate difficulty, neither easy nor tough.",
        ],
    },
    "Facilities": {
        "Positive": [
            "The library has a huge collection of books and a peaceful reading area.",
            "Computer labs are well equipped with latest systems and good internet.",
            "The campus is clean, green and feels very safe.",
            "Sports ground and gym are well maintained and open till late evening.",
        ],
        "Negative": [
            "The wifi in the hostel is very slow and keeps disconnecting.",
            "Classrooms are overcrowded and the fans do not work properly.",
            "Canteen food is costly and the quality is bad.",
            "There is no proper drinking water and the washrooms are dirty.",
        ],
        "Neutral": [
            "The library opens at nine in the morning and closes at eight in the evening.",
            "The canteen is located near the main block.",
            "There are three computer labs on the second floor.",
            "The hostel and the sports ground are on the other side of the campus.",
        ],
    },
    "Administration": {
        "Positive": [
            "The office staff replied to my queries quickly and politely.",
            "The scholarship process was smooth and the staff guided me at every step.",
            "Admission was hassle free and the online portal is easy to use.",
            "The placement cell conducts regular training sessions and keeps us informed.",
        ],
        "Negative": [
            "I had to visit the office many times just to get a simple certificate.",
            "The fee payment portal keeps crashing and nobody responds to complaints.",
            "Timetable changes are announced at the last minute without any notice.",
            "The administration does not listen to student problems.",
        ],
        "Neutral": [
            "The office is open from nine to four on all working days.",
            "Fees can be paid online or at the bank counter.",
            "Notices are posted on the department notice board and the website.",
            "The administration handles most requests within a few days.",
        ],
    },
}


def add_adverb(adj, sentiment):
    """Randomly put an intensifier in front of an adjective when it sounds natural."""
    if adj.split()[0] in NO_ADVERB_FIRST:
        return adj
    options = ["", "", "really ", "very ", "quite "] if sentiment != "Neutral" else ["", "", "fairly ", "quite "]
    return random.choice(options) + adj


def add_typo(text):
    """Swap two neighbouring letters in one longer word (simulates a typing mistake)."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 5 and w.isalpha()]
    if not candidates:
        return text
    i = random.choice(candidates)
    w = words[i]
    j = random.randint(1, len(w) - 3)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def make_comment(aspect, sentiment):
    cfg = ASPECTS[aspect]
    noun, be = random.choice(cfg["nouns"])

    if sentiment == "Positive":
        adj = add_adverb(random.choice(cfg["pos"]), sentiment)
        tail = random.choice(POS_TAILS)
        frame = random.choice(FRAMES)
    elif sentiment == "Negative":
        tail = random.choice(NEG_TAILS)
        if random.random() < 0.22:  # negated positive word, e.g. "not helpful"
            adj = random.choice(cfg["negpos"])
            frame = random.choice(NEGATION_FRAMES)
        else:
            adj = add_adverb(random.choice(cfg["neg"]), sentiment)
            frame = random.choice(FRAMES)
    else:
        adj = add_adverb(random.choice(NEUTRAL_ADJ), sentiment)
        tail = random.choice(NEU_TAILS)
        frame = random.choice(FRAMES)

    text = frame.format(n=noun, be=be, adj=adj, tail=tail)

    # a little noise so the data is not perfectly clean
    if random.random() < 0.07:
        text = add_typo(text)
    if random.random() < 0.08:
        text = text.lower().rstrip(".")
    return text


def main():
    rows = []
    seen = set()

    # 1) hand-written comments
    for aspect, by_sent in HAND.items():
        for sentiment, comments in by_sent.items():
            for c in comments:
                rows.append((c, aspect, sentiment))
                seen.add(c.lower())

    # 2) template-generated comments
    # (positive, negative, neutral) comments to generate per aspect - different on
    # purpose, like real feedback: teaching is liked, facilities are criticised.
    targets_by_aspect = {
        "Teaching":       {"Positive": 34, "Negative": 14, "Neutral": 16},
        "Course Content": {"Positive": 22, "Negative": 26, "Neutral": 16},
        "Exams":          {"Positive": 18, "Negative": 30, "Neutral": 16},
        "Facilities":     {"Positive": 14, "Negative": 36, "Neutral": 14},
        "Administration": {"Positive": 24, "Negative": 26, "Neutral": 14},
    }
    for aspect in ASPECTS:
        for sentiment, target in targets_by_aspect[aspect].items():
            made, attempts = 0, 0
            while made < target and attempts < 5000:
                attempts += 1
                c = make_comment(aspect, sentiment)
                if c.lower() in seen:
                    continue
                seen.add(c.lower())
                rows.append((c, aspect, sentiment))
                made += 1

    random.shuffle(rows)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "feedback", "aspect", "sentiment"])
        for i, (c, a, s) in enumerate(rows, start=1):
            w.writerow([i, c, a, s])

    print(f"Saved {len(rows)} comments to {OUT_FILE}")
    for s in ("Positive", "Neutral", "Negative"):
        print(f"  {s:9s}: {sum(1 for r in rows if r[2] == s)}")


if __name__ == "__main__":
    main()
