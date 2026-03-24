import random

# -----------------------------
# Joint Distribution (EDIT THIS)
# -----------------------------
joint = {
    ("sunny", "hot"): 0.45,
    ("sunny", "cold"): 0.15,
    ("rain", "hot"): 0.02,
    ("rain", "cold"): 0.08,
    ("fog", "hot"): 0.03,
    ("fog", "cold"): 0.27,
    ("meteor", "hot"): 0.00,
    ("meteor", "cold"): 0.00,
}

# Normalize (just in case)
total = sum(joint.values())
joint = {k: v / total for k, v in joint.items()}


# -----------------------------
# Sampling from joint
# -----------------------------
def sample_joint(joint_dist):
    outcomes = list(joint_dist.keys())
    probs = list(joint_dist.values())
    return random.choices(outcomes, weights=probs, k=1)[0]


# -----------------------------
# Probability helpers
# -----------------------------
def prob_event(joint_dist, condition_fn):
    return sum(p for (w, t), p in joint_dist.items() if condition_fn(w, t))


def conditional_prob(joint_dist, event_fn, given_fn):
    numerator = prob_event(joint_dist, lambda w, t: event_fn(w, t) and given_fn(w, t))
    denominator = prob_event(joint_dist, given_fn)
    return numerator / denominator if denominator > 0 else 0


# -----------------------------
# Game setup and play loop
# -----------------------------
def build_rounds(joint_dist):
    return [
        {
            "left_label": "Sun",
            "right_label": "Hot",
            "left_prob": prob_event(joint_dist, lambda w, t: w == "sunny"),
            "right_prob": prob_event(joint_dist, lambda w, t: t == "hot"),
        },
        {
            "left_label": "Cold and Fog",
            "right_label": "Rain",
            "left_prob": prob_event(joint_dist, lambda w, t: w == "fog" and t == "cold"),
            "right_prob": prob_event(joint_dist, lambda w, t: w == "rain"),
        },
        {
            "left_label": "Cold | Rain (P(cold | rain))",
            "right_label": "Sunny",
            "left_prob": conditional_prob(
                joint_dist,
                lambda w, t: t == "cold",
                lambda w, t: w == "rain",
            ),
            "right_prob": prob_event(joint_dist, lambda w, t: w == "sunny"),
        },
        {
            "left_label": "Foggy | Cold",
            "right_label": "Hot | not Rain",
            "left_prob": conditional_prob(
                joint_dist,
                lambda w, t: w == "fog",
                lambda w, t: t == "cold",
            ),
            "right_prob": conditional_prob(
                joint_dist,
                lambda w, t: t == "hot",
                lambda w, t: w != "rain",
            ),
        },
        {
            "left_label": "Hot, then Sun (P(sunny | hot))",
            "right_label": "Hot and Sun",
            "left_prob": conditional_prob(
                joint_dist,
                lambda w, t: w == "sunny",
                lambda w, t: t == "hot",
            ),
            "right_prob": prob_event(joint_dist, lambda w, t: w == "sunny" and t == "hot"),
        },
    ]


def ask_guess():
    while True:
        choice = input("Pick 1 or 2 (or q to quit): ").strip().lower()
        if choice in {"1", "2", "q"}:
            return choice
        print("Please enter 1, 2, or q.")


def play_game(joint_dist):
    rounds = build_rounds(joint_dist)
    score = 0

    print("---- Probability Guessing Game ----")
    print("Guess which outcome will be selected each round.\n")

    for i, round_data in enumerate(rounds, start=1):
        left_label = round_data["left_label"]
        right_label = round_data["right_label"]
        left_prob = round_data["left_prob"]
        right_prob = round_data["right_prob"]

        print(f"Round {i}:")
        print(f"(1) {left_label}\t\t(2) {right_label}")

        guess = ask_guess()
        if guess == "q":
            print("\nGame ended early.")
            break

        # Pick a result weighted by each outcome's probability.
        result_index = random.choices([1, 2], weights=[left_prob, right_prob], k=1)[0]
        result_label = left_label if result_index == 1 else right_label
        guessed_right = int(guess) == result_index

        if guessed_right:
            score += 1

        print(f"P1={left_prob:.3f}, P2={right_prob:.3f}")
        print(f"Weighted draw winner: Choice ({result_index}) {result_label}")
        print("You win!\n" if guessed_right else "Not this time.\n")

    print(f"Final score: {score}/{len(rounds)}")


# -----------------------------
# Run everything
# -----------------------------
if __name__ == "__main__":
    play_game(joint)