import random

# -----------------------------
# Marginal distribution P(Temperature)
# -----------------------------
marginal_temp = {
    "hot": 0.25,
    "cold": 0.75,
}

# -----------------------------
# Conditional distribution P(Weather | Temperature)
# Each row must sum to 1 for a given temperature.
# We normalize per temperature slice for safety.
# -----------------------------
cond_weather_given_temp = {
    ("sunny", "hot"): 0.90,
    ("sunny", "cold"): 0.30,
    ("rain",  "hot"): 0.04,
    ("rain",  "cold"): 0.16,
    ("fog",   "hot"): 0.06,
    ("fog",   "cold"): 0.54,
    ("meteor","hot"): 0.00,
    ("meteor","cold"): 0.00,
}

# Normalize each temperature slice so each sums to 1
for temp in marginal_temp:
    slice_total = sum(v for (w, t), v in cond_weather_given_temp.items() if t == temp)
    if slice_total > 0:
        cond_weather_given_temp = {
            (w, t): (v / slice_total if t == temp else v)
            for (w, t), v in cond_weather_given_temp.items()
        }

# -----------------------------
# Product Rule: P(W, T) = P(W | T) * P(T)
# -----------------------------
joint = {
    (w, t): cond_weather_given_temp[(w, t)] * marginal_temp[t]
    for (w, t) in cond_weather_given_temp
}

# Normalize joint (just in case of floating-point drift)
_jt = sum(joint.values())
joint = {k: v / _jt for k, v in joint.items()}


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
            "left_label":    "Sun",
            "right_label":   "Fog",
            "left_prob":  prob_event(joint_dist, lambda w, t: w == "sunny"),
            "right_prob": prob_event(joint_dist, lambda w, t: w == "fog"),
            "left_formula":  "P(sunny) = sum over all temps of P(sunny, t)",
            "right_formula": "P(fog)   = sum over all temps of P(fog, t)",
        },
        {
            "left_label":    "Sun and Hot",
            "right_label":   "Fog and Cold",
            "left_prob":  prob_event(joint_dist, lambda w, t: w == "sunny" and t == "hot"),
            "right_prob": prob_event(joint_dist, lambda w, t: w == "fog"   and t == "cold"),
            "left_formula":  "P(sunny, hot) = P(sunny|hot) * P(hot)",
            "right_formula": "P(fog, cold)  = P(fog|cold)  * P(cold)",
        },
        {
            "left_label":    "Hot | Sun",
            "right_label":   "Cold | Sun",
            "left_prob": conditional_prob(
                joint_dist,
                lambda w, t: t == "hot",
                lambda w, t: w == "sunny",
            ),
            "right_prob": conditional_prob(
                joint_dist,
                lambda w, t: t == "cold",
                lambda w, t: w == "sunny",
            ),
            "left_formula":  "P(hot|sunny)  = P(sunny, hot)  / P(sunny)",
            "right_formula": "P(cold|sunny) = P(sunny, cold) / P(sunny)",
        },
        {
            "left_label":    "Hot | not Fog",
            "right_label":   "Cold | not Fog",
            "left_prob": conditional_prob(
                joint_dist,
                lambda w, t: t == "hot",
                lambda w, t: w != "fog",
            ),
            "right_prob": conditional_prob(
                joint_dist,
                lambda w, t: t == "cold",
                lambda w, t: w != "fog",
            ),
            "left_formula":  "P(hot|~fog)  = P(~fog, hot)  / P(~fog)",
            "right_formula": "P(cold|~fog) = P(~fog, cold) / P(~fog)",
        },
        {
            "left_label":    "Sun | Hot",
            "right_label":   "Sun | Cold",
            "left_prob": conditional_prob(
                joint_dist,
                lambda w, t: w == "sunny",
                lambda w, t: t == "hot",
            ),
            "right_prob": conditional_prob(
                joint_dist,
                lambda w, t: w == "sunny",
                lambda w, t: t == "cold",
            ),
            "left_formula":  "P(sunny|hot)  = P(sunny, hot)  / P(hot)",
            "right_formula": "P(sunny|cold) = P(sunny, cold) / P(cold)",
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

    print("=== Product Rule Probability Guessing Game ===")
    print("Probabilities are derived using the product rule:")
    print("  P(Weather, Temp) = P(Weather | Temp) * P(Temp)\n")
    print("Guess which outcome will be randomly selected each round.\n")

    for i, round_data in enumerate(rounds, start=1):
        left_label  = round_data["left_label"]
        right_label = round_data["right_label"]
        left_prob   = round_data["left_prob"]
        right_prob  = round_data["right_prob"]

        print(f"Round {i}:")
        print(f"  (1) {left_label}   vs   (2) {right_label}")

        guess = ask_guess()
        if guess == "q":
            print("\nGame ended early.")
            break

        # Pick a result weighted by each outcome's probability
        result_index = random.choices([1, 2], weights=[left_prob, right_prob], k=1)[0]
        result_label = left_label if result_index == 1 else right_label
        guessed_right = int(guess) == result_index

        if guessed_right:
            score += 1

        print(f"  Formula: {round_data['left_formula' if result_index == 1 else 'right_formula']}")
        print(f"  P1 = {left_prob:.4f}   P2 = {right_prob:.4f}")
        print(f"  Weighted draw outcome:  ({result_index}) {result_label}")
        print("  You win!\n" if guessed_right else "  Not this time.\n")

    print(f"Final score: {score}/{len(rounds)}")


# -----------------------------
# Run everything
# -----------------------------
if __name__ == "__main__":
    play_game(joint)
