"""
smart_agent.py  —  Single-Agent Smart Assistant
Tools: Calculator, Keyword Extractor, Unit Converter, Summariser
"""

import re
import math
import json
import datetime
import os

# ── Create logs folder ────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)


# ═══════════════════════════════════════════════════
# TOOL 1 — CALCULATOR
# ═══════════════════════════════════════════════════
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        expr = expression.strip()

        # Handle '15% of 200' pattern
        pct = re.match(r"([\d.]+)%\s+of\s+([\d.]+)", expr)
        if pct:
            result = float(pct.group(1)) / 100 * float(pct.group(2))
            return str(int(result)) if result == int(result) else str(round(result, 4))

        # Keep only safe characters (digits, operators, math function names)
        expr = re.sub(r"[^0-9a-zA-Z+\-*/().% \t^]", "", expr)
        expr = expr.replace("^", "**")   # support ^ for power

        # Safe environment — only math functions allowed
        safe_env = {
            "sqrt" : math.sqrt,
            "sin"  : math.sin,
            "cos"  : math.cos,
            "tan"  : math.tan,
            "log"  : math.log,
            "log10": math.log10,
            "abs"  : abs,
            "pi"   : math.pi,
            "e"    : math.e,
            "__builtins__": {},   # blocks any dangerous Python built-ins
        }

        result = eval(expr, safe_env)

        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        elif isinstance(result, float):
            return str(round(result, 4))
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"


# ═══════════════════════════════════════════════════
# TOOL 2 — KEYWORD EXTRACTOR
# ═══════════════════════════════════════════════════
# Common words that carry no meaning
STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall",
    "should","may","might","must","can","could","to","of","in",
    "on","at","by","for","with","about","from","into","through",
    "this","that","these","those","it","its","we","our","they",
    "their","what","which","who","when","where","how","and","or",
    "but","not","so","if","as","than","then","also","just","very",
}

def extract_keywords(text: str) -> list:
    """Extract meaningful keywords from text."""
    try:
        # Get all words (letters and numbers only)
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())

        # Remove short words and stop words
        filtered = [w for w in words if len(w) > 3 and w not in STOP_WORDS]

        # Count frequency of each word
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        # Sort by frequency (most common first), then alphabetically
        ranked = sorted(freq.keys(), key=lambda w: (-freq[w], w))

        return ranked[:7]   # return top 7 keywords
    except Exception:
        return []


# ═══════════════════════════════════════════════════
# TOOL 3 — UNIT CONVERTER  (bonus)
# ═══════════════════════════════════════════════════
CONVERSIONS = {
    ("km",    "miles")      : 0.621371,
    ("miles", "km")         : 1.60934,
    ("m",     "ft")         : 3.28084,
    ("ft",    "m")          : 0.3048,
    ("cm",    "inch")       : 0.393701,
    ("inch",  "cm")         : 2.54,
    ("kg",    "lbs")        : 2.20462,
    ("lbs",   "kg")         : 0.453592,
    ("g",     "oz")         : 0.035274,
    ("oz",    "g")          : 28.3495,
    ("kmph",  "mph")        : 0.621371,
    ("mph",   "kmph")       : 1.60934,
}

def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units of measurement."""
    try:
        f = from_unit.lower().strip()
        t = to_unit.lower().strip()

        # Temperature — special formula
        if f == "celsius" and t == "fahrenheit":
            result = value * 9/5 + 32
            return f"{value}°C = {round(result, 2)}°F"
        if f == "fahrenheit" and t == "celsius":
            result = (value - 32) * 5/9
            return f"{value}°F = {round(result, 2)}°C"

        # All other conversions
        if (f, t) in CONVERSIONS:
            result = value * CONVERSIONS[(f, t)]
            return f"{value} {f} = {round(result, 4)} {t}"

        return f"Conversion from {f} to {t} is not supported."
    except Exception as e:
        return f"Conversion error: {str(e)}"


# ═══════════════════════════════════════════════════
# TOOL 4 — TEXT SUMMARISER  (bonus)
# ═══════════════════════════════════════════════════
def summarise_text(text: str) -> str:
    """Pick the 2 most important sentences from a paragraph."""
    try:
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        if len(sentences) <= 2:
            return text.strip()

        # Score each sentence by how many keywords it contains
        keywords = set(extract_keywords(text))
        def score(sentence):
            words = re.findall(r"[a-z]+", sentence.lower())
            return sum(1 for w in words if w in keywords)

        # Pick top 2 sentences, keeping original order
        ranked = sorted(sentences, key=score, reverse=True)
        top_2  = sorted(ranked[:2], key=lambda s: sentences.index(s))
        return " ".join(top_2)
    except Exception:
        return text[:200]


# ═══════════════════════════════════════════════════
# INTENT CLASSIFIER
# Reads the query and decides which tool to use
# ═══════════════════════════════════════════════════
def classify_intent(query: str) -> str:
    """Return the best-matching intent for a query."""
    q = query.lower()

    # Check for calculation patterns
    calc_patterns = [
        r"\bcalculate\b", r"\bcompute\b", r"\bsolve\b",
        r"\bwhat is\b.*\d",          # "what is 2 + 2"
        r"\d(?:\s*[+\-*/^]+\s*)\d", # "5 + 3", "12 * 4", or "2 ** 8"
        r"\bsqrt\b", r"\bsquare root\b", r"\bpercent\b",
    ]
    if any(re.search(p, q) for p in calc_patterns):
        return "calculation"

    # Check for keyword extraction
    kw_patterns = [r"\bkeyword", r"\bextract\b"]
    if any(re.search(p, q) for p in kw_patterns):
        return "keywords"

    # Check for unit conversion
    conv_patterns = [
        r"\bconvert\b",
        r"\bto\s+(km|miles|kg|lbs|celsius|fahrenheit|mph|kmph|ft|inch|cm)\b",
    ]
    if any(re.search(p, q) for p in conv_patterns):
        return "conversion"

    # Check for summarisation
    summ_patterns = [r"\bsummar", r"\btldr\b", r"\bbriefly\b"]
    if any(re.search(p, q) for p in summ_patterns):
        return "summary"

    # Default: general response
    return "general"


# ═══════════════════════════════════════════════════
# SHORT QUIZ MODE
# ═══════════════════════════════════════════════════
def run_quiz(questions=None, answers=None, interactive=True):
    """Run a short quiz and return a score summary."""
    if questions is None:
        questions = [
            ("What is 2 + 2?", "4"),
            ("What is 10 - 3?", "7"),
            ("What is 4 * 5?", "20"),
        ]
    if answers is None:
        answers = []

    print("\n=== Short Quiz ===")
    print("Answer each question and press Enter.")

    score = 0
    results = []

    for idx, (prompt, expected) in enumerate(questions, start=1):
        if interactive:
            user_answer = input(f"{idx}. {prompt} ").strip()
        else:
            user_answer = answers[idx - 1] if idx - 1 < len(answers) else ""

        correct = str(user_answer).strip().lower() == str(expected).strip().lower()
        if correct:
            score += 1
            status = "correct"
        else:
            status = "incorrect"

        results.append({
            "question": prompt,
            "expected": expected,
            "answer": user_answer,
            "correct": correct,
            "status": status,
        })

    percent = round((score / len(questions)) * 100, 1) if questions else 0.0
    print(f"\nQuiz complete! You scored {score}/{len(questions)} ({percent}%).")

    return {
        "type": "quiz",
        "score": score,
        "max_score": len(questions),
        "percentage": percent,
        "results": results,
    }


# ═══════════════════════════════════════════════════
# GENERAL RESPONSE
# Handles common questions without a tool
# ═══════════════════════════════════════════════════
GENERAL_ANSWERS = {
    r"\bwhat is (machine learning|ml)\b":
        "Machine Learning is a subset of AI where computers learn patterns "
        "from data to make predictions — without being explicitly programmed.",
    r"\bwhat is (deep learning|dl)\b":
        "Deep Learning uses multi-layered neural networks to learn complex "
        "patterns from large amounts of data.",
    r"\bwhat is (artificial intelligence|ai)\b":
        "Artificial Intelligence is the ability of a machine to simulate "
        "human intelligence — including learning, reasoning, and problem-solving.",
    r"\bwhat is (nlp|natural language processing)\b":
        "NLP (Natural Language Processing) is the branch of AI that helps "
        "computers understand, interpret, and generate human language.",
    r"\bwhat is python\b":
        "Python is a high-level programming language known for its simple "
        "syntax, readability, and huge library ecosystem — ideal for AI/ML.",
    r"\bhello|hi|hey\b":
        "Hello! I'm your Smart Assistant. I can calculate, extract keywords, "
        "convert units, and summarise text. Type 'help' to see examples.",
    r"\bhelp\b":
        "I can help with:\n"
        "  • Calculations     →  Calculate 25 * 4 + 10\n"
        "  • Keywords         →  Extract keywords from Deep learning is amazing\n"
        "  • Unit conversion  →  Convert 100 km to miles\n"
        "  • Summaries        →  Summarise: <your text here>\n"
        "  • General Q&A      →  What is machine learning?",
}

def general_response(query: str) -> str:
    """Return a helpful response for general questions."""
    q = query.lower()
    for pattern, answer in GENERAL_ANSWERS.items():
        if re.search(pattern, q):
            return answer
    return (
        f"I received your query: \"{query}\".\n"
        "Try: 'calculate ...', 'extract keywords from ...', "
        "'convert X unit to unit', or 'summarise ...'."
    )


# ═══════════════════════════════════════════════════
# HELPER: pull expression out of sentence
# ═══════════════════════════════════════════════════
def strip_intent_words(query: str) -> str:
    """Remove instruction words, leaving just the content."""
    cleaned = re.sub(
        r"^(calculate|compute|solve|evaluate|what is|find|work out)"
        r"[:\s]*", "", query, flags=re.IGNORECASE
    ).strip()
    return cleaned.rstrip("?.,!") or query


# ═══════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ═══════════════════════════════════════════════════
def agent(query: str) -> dict:
    """
    Single-Agent Smart Assistant.
    Takes a natural language query.
    Returns a structured dictionary (JSON-compatible).
    """
    # Handle empty input
    if not query or not query.strip():
        return {
            "type"     : "error",
            "result"   : "Empty query. Please enter a question.",
            "tool"     : None,
            "query"    : query,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # Step 1: Classify intent
    intent = classify_intent(query)

    try:
        # ── Route to Calculator ───────────────────────────────
        if intent == "calculation":
            expression = strip_intent_words(query)
            result     = calculator(expression)
            return {
                "type"      : "calculation",
                "expression": expression,
                "result"    : result,
                "tool"      : "calculator",
                "query"     : query,
                "timestamp" : datetime.datetime.now().isoformat(),
            }

        # ── Route to Keyword Extractor ────────────────────────
        elif intent == "keywords":
            text = re.sub(
                r"^(extract keywords from|keywords from|"
                r"find keywords in|keywords:)[:\s]*",
                "", query, flags=re.IGNORECASE
            ).strip()
            keywords = extract_keywords(text)
            return {
                "type"      : "keywords",
                "input_text": text,
                "result"    : keywords,
                "count"     : len(keywords),
                "tool"      : "keyword_extractor",
                "query"     : query,
                "timestamp" : datetime.datetime.now().isoformat(),
            }

        # ── Route to Unit Converter ───────────────────────────
        elif intent == "conversion":
            # Parse: "convert 100 km to miles" or "37 celsius to fahrenheit"
            match = re.search(
                r"([\d.]+)\s*([a-zA-Z°/]+)\s+to\s+([a-zA-Z°/]+)",
                query, re.IGNORECASE
            )
            if match:
                value     = float(match.group(1))
                from_unit = match.group(2)
                to_unit   = match.group(3)
                result    = convert_units(value, from_unit, to_unit)
                return {
                    "type"     : "conversion",
                    "result"   : result,
                    "tool"     : "unit_converter",
                    "query"    : query,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            else:
                return {
                    "type"     : "error",
                    "result"   : "Could not parse. Try: 'Convert 100 km to miles'",
                    "tool"     : None,
                    "query"    : query,
                    "timestamp": datetime.datetime.now().isoformat(),
                }

        # ── Route to Summariser ───────────────────────────────
        elif intent == "summary":
            text = re.sub(
                r"^(summarise|summarize|summary of|briefly|tldr)[:\s]*",
                "", query, flags=re.IGNORECASE
            ).strip()
            result = summarise_text(text)
            return {
                "type"      : "summary",
                "input_text": text[:80] + "..." if len(text) > 80 else text,
                "result"    : result,
                "tool"      : "summariser",
                "query"     : query,
                "timestamp" : datetime.datetime.now().isoformat(),
            }

        # ── General response ──────────────────────────────────
        else:
            return {
                "type"     : "general",
                "result"   : general_response(query),
                "tool"     : None,
                "query"    : query,
                "timestamp": datetime.datetime.now().isoformat(),
            }

    except Exception as e:
        return {
            "type"     : "error",
            "result"   : f"Agent error: {str(e)}",
            "tool"     : None,
            "query"    : query,
            "timestamp": datetime.datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════
# REQUIRED TEST CASES (from your project brief)
# ═══════════════════════════════════════════════════
def run_required_tests():
    """Run the 3 test cases from the project brief."""
    queries = [
        "Calculate 20 + 5",
        "Extract keywords from Artificial Intelligence is transforming industries",
        "What is machine learning?"
    ]

    print("\n" + "=" * 55)
    print("  REQUIRED TEST CASES")
    print("=" * 55)

    for q in queries:
        print(f"\nQuery   : {q}")
        print(f"Response: {agent(q)}")
        print("-" * 55)


# ═══════════════════════════════════════════════════
# BONUS TESTS
# ═══════════════════════════════════════════════════
def run_all_tests():
    """Run all test cases including bonus tools."""
    all_queries = [
        # Required
        "Calculate 20 + 5",
        "Extract keywords from Artificial Intelligence is transforming industries",
        "What is machine learning?",
        # More calculator tests
        "Calculate 15 * 4 + 10",
        "Compute 100 / 4",
        "What is 2 ** 8",
        "Calculate sqrt(144)",
        "Solve 15% of 200",
        # More keyword tests
        "Extract keywords from Deep learning uses neural networks to process data",
        # Unit conversion (bonus)
        "Convert 100 km to miles",
        "Convert 37 celsius to fahrenheit",
        "Convert 75 kg to lbs",
        # Summariser (bonus)
        "Summarise: Machine learning is a type of AI. It allows computers to "
        "learn from data. Models improve with experience.",
        # General
        "What is Python?",
        "hello",
        "help",
    ]

    print("\n" + "=" * 55)
    print("  ALL TEST CASES")
    print("=" * 55)

    passed = 0
    for q in all_queries:
        result = agent(q)
        status = "✓" if result["type"] != "error" else "✗"
        if result["type"] != "error":
            passed += 1
        short_q = q[:52] + "..." if len(q) > 52 else q
        res_str = str(result["result"])[:70]
        print(f"\n  {status} [{result['type']:12s}] {short_q}")
        print(f"     Result: {res_str}")

    print(f"\n{'=' * 55}")
    print(f"  Passed: {passed}/{len(all_queries)}")
    print(f"{'=' * 55}\n")


# ═══════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════
def interactive():
    """Let you type queries one by one."""
    print("\n" + "=" * 55)
    print("  Smart Agent — Interactive Mode")
    print("  Type 'exit' to quit | Type 'help' for examples")
    print("=" * 55)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if user_input.lower() in ("exit", "quit", "bye"):
            print("  Goodbye!")
            break

        if not user_input:
            continue

        result = agent(user_input)
        print(f"\n  Response:\n{json.dumps(result, indent=2)}")


# ═══════════════════════════════════════════════════
# ENTRY POINT — runs when you do: python smart_agent.py
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1: Show required test cases
    run_required_tests()

    # Step 2: Show all tests including bonus
    run_all_tests()

    # Step 3: Choose mode
    print("\nChoose a mode:")
    print("  1. Agent chat")
    print("  2. Short quiz")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        quiz_result = run_quiz(interactive=True)
        print("\nQuiz complete")
        print(f"Score: {quiz_result['score']}/{quiz_result['max_score']}")
    else:
        interactive()