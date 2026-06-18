# Fake News Headline Detector — UI

A polished **Next.js frontend** for a C++-powered NLP system that evaluates news headlines and visualizes fake-likelihood using calibrated, interpretable UI design.

🔗 **Live Demo:** https://nlp-fake-news-headline-detector-ui.vercel.app/
**Please be patient for first query as the render server takes ≈ 1 minute to startup, queries after boot occur in 1-2 seconds**

## Demo

![NLP Fake News Detector UI](NLP%20Fake%20News%20Detector.png)

---

## Overview

This interface allows users to input a news headline and receive a **fake-likelihood score** displayed on a real-to-fake spectrum, along with a natural-language interpretation of the result.

The goal of this project is **not** to declare truth or intent, but to demonstrate how machine learning outputs can be communicated **responsibly and clearly** to users.

---

## How It Works

1. A user enters a news headline
2. The frontend sends the headline to a FastAPI backend
3. A C++ NLP model returns a continuous fake-likelihood score
4. The UI maps this score to:
   - A visual real ↔ fake spectrum
   - A calibrated natural-language prediction
   - Clear uncertainty cues

---

## Tech Stack

**Frontend**
- Next.js (App Router)
- TypeScript
- Tailwind CSS

**Backend**
- FastAPI
- C++

---

## Design Principles

- Avoid misleading “confidence percentages”
- Treat ML output as **probabilistic signals**, not facts
- Make uncertainty visible and intuitive
- Prioritize clarity over false authority

---

## Important Note

This project analyzes **linguistic patterns**, not factual correctness.  
The output reflects similarity to patterns commonly seen in fake news headlines and **should not be interpreted as a statement of truth, intent, or credibility**.

---

## Related Repository

- **Backend / API:** https://github.com/NeeleshN1/fake-news-headline-detector-api

---

## Author

Built by **Neelesh Nayak** as a full-stack NLP portfolio project.
