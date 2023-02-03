---
colorlinks: true
geometry: margin=2cm
---

## Semeval 2023 Task 2: [Multiconer 2](https://multiconer.github.io/)
Briefly, MultiCoNER it is an NER task that requires identifying entities in a sentence and classifying them into **35** 
fine-grained types such as `MusicalWork`, `Medication`, and `Drink`.

### Types of Models used
- **Sequence Labeling**
  - Bert Layer representing tokens.
  - Linear Classifier Layer labels each token with BIO scheme.
- **Sequence Labeling CRF**
  - Bert Layer representing tokens.
  - A CRF Classifier Layer labels each token with BIO labels. 
- **Span** (implementation of [SpanNER][spanner])
  - Bert Layer representing tokens.
  - Each Span is represented by concatenating the first and the last token embedding.
  - Linear Classifier Layer labels each span with BIO labels.
- **Span with Span length**
  - Same as **Span**, but with span length representations concatenated   
    to span representations.
- **Span with Noun Phrase**
  - Same as **Span**, but with noun-phrase representations(one-hot) concatenated 
    to span representations.

### Findings
#### Large language models were significantly better
Below are results(all F1 scores) comparing the use of `bert-base` and `xlm-roberta-large`:

| Seq Label Base | Span Base | Seq Label Large | Span Large |
|---|---|---|---|
| 0.45 | 0.47 | 0.65 | 0.65 |

The large models score around 0.20 more F1 points (44% higher). Moreover, note that
the large span model performs the same as the large sequence labeling model.
However, when base models are used, the span model performs better. 

#### Degradation in performance when I included noun-phrase information
After identifying all spans that correspond to noun-phrases using the [Benepar Constituency Parser][benepar],
I incorporated noun-phrase information into the span representations
by concatenating a binary value(presence or absence of a noun-phrase) to them. I saw a degradation which is
illustrated below:

| Seq Label Base | Span Base | Span Noun Phrase Base |
|---|---|---------|
| 0.45 | 0.47 | **0.42**|

This analysis is currently restricted to smaller language models.

#### Identifying invalid span predictions
In the span approach, every possible span is labeled, so it is possible to predict overlapping spans. However,
the MultiCoNER task doesn't allow overlapping spans. Therefore, I used a heuristic (also used in 
the SpanNER [paper][spanner]) to remove all overlapping spans. In short, the algorithm is:
> If `s` is a span that overlaps with other spans, remove `s` if 
> the model predicts any of the other overlapping spans with a higher probabiliy.

I was expecting to see an improvement, but saw none (see table below):

| Span-With-Heuristic | Span |
|---|---|
| 0.654 | 0.655 |

This is surprising because I did see several overlapping spans being predicted ; 
more analysis is required.

#### Using a CRF(Conditional Random Field) layer to classify tokens hurt performance:
The [Damo NLP team][damo] (winners of last year's Multiconer) swear by CRF for sequence labeling. However, 
when I tried using CRF instead of a linear classifier, the performance degraded as follows:

| Span-Base | Sequence-Base | Sequence-CRF-Base |
|---|---|-------------------|
| 0.47 | 0.45 | **0.37**          |

Interestingly, CRF performs much better on the [LegalEval][legal] SemEval Task:

| Dataset | Sequence-Base | Span-Base | Sequence-CRF-Base |
|---|---|---|-------------------|
| LegalEval-Judgement | 0.85 | 0.86 | **0.86**          |
| LegalEval-Preamble | 0.79 | 0.69 | **0.77**          |
| Multiconer | 0.45 | 0.47 | _0.37_            |

#### Adding span-size information to spans didn't improve performance.
[SpanNER][spanner] claims that adding span length information to spans can increase performance
, but I didn't see an improvement:

| Span-Large | Span-Large-with-Span-Size-Info | Sequence-Large |
|---|---|---|
| 0.65 | 0.64 | 0.65 |

Like the [SpanNER][spanner] paper authors, 
I represented span sizes using learnable vectors, 
which I later are concatenated to the span representations. However,
I don't think this is the best approach to this problem.

### Submission
I submitted the following 4 models:
 - **SpanLarge**: [SpanNER][spanner] with _XLM Roberta Large_ embeddings
 - **SeqLarge**: Sequence Labeling model with _XML Roberta Large_ embeddings
 - **SpanLarge With Heuristic**: Same as **SpanLarge** but with a heuristic 
that removes overlapping spans
 - **SpanLarge With Span Length**: Same as **SpanLarge** but with added span-length information to span representations.

[legal]: https://sites.google.com/view/legaleval/home
[damo]: https://aclanthology.org/2022.semeval-1.200/
[spanner]:https://arxiv.org/abs/2106.00641 
[benepar]:https://arxiv.org/abs/1805.01052
