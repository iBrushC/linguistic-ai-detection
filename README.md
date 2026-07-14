# Linguistic AI Detection
This repo uses linguistic stylometry with human and AI authors to explore the question of detecting AI by inversion, meaning by detecting an individual human author rather than general "AI".

## Why
It is widely known that "AI detectors" are extremely flawed, many of them being overly or underly sensitive. The problem is nearly impossible due to the sheer amount of different LLMs, how many personalize their conversation style to the user, and that a single essay (or worse paragraph) just cannot easily be attributed to anyone.

Despite that, there are many cases where it would be useful to detect AI written works, primarily in education. There's a growing body of research showing the negative side effects of over-reliance on AI, so reliably identifying it could be very useful. Since detecting the ambiguous "AI" is a near impossible challenge, what if we tried to detect individual authors and identify by inversion?

## How
The basis is linguistic stylometry, something which has long been used for author attribution (some famous works include *The Two Noble Kinsmen and Henry VIII*). Using various computed metrics, we can compare an unknown piece of text to a known corpus. If the statistics of the new text match with the corpus, we consider it written by the author. We create a combined metric using a number of factors (detailed below) and set a threshold for passing or failing the test.

What we are exactly testing in this package is if it could be viable for isolating an author from an AI, even if that AI is trying its very best to copy the author. 

### The Tests
There are three tests to push the stylometry-based detection as far as possible

1. Author vs Author: how accurately does the metric detect between human authors?
2. Author vs AI: how accurately does the metric detect between a human author and an AI trying to recreate one of their writings?
3. Author vs Knowledgeable AI: how accurately does the metric detect between a human author and an AI rewriting one of their works with in-depth knowledge of every single statistic being tested?

### Texts Used
For the ground truth author-attribution, public domain essays are used, all from before 2023. The authors of the essays used are Ross Bullen, Frank Delaney, Erica X Eisen, Matthew Green, Christine Jones. Ross Bullen has four essays while the others have three essays each.

## AIs Tested
We test the most popular LLMs, two closed and two open:
* ChatGPT 5.6
* Claude Opus 4.8
* Deepseek V4 Pro
* GLM 5.2