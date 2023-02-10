---
colorlinks: true
geometry: margin=2cm
---

# Goal: learn to segment text
- Framed as a Token Classification problem:
  - External Knowledge:
    - Concatenating Wikipedia/Google search results.
    - [Transformer Encoder][transformer_ext].
    - [MiRIM][mi-rim]: Recurrent Independent Mechanisms.
    - DAM: Dynamic Assorted Modules.
  - Using context better:
    - [FLERT][flert] approach
- Framed as a Span Classification problem:
  - External Knowledge:
    - Concatenating features to span reps vs token reps.
  - Using context better:
    - [FLERT][flert] for spans ?
- Framed as a Machine Reading Comprehension problem:
  - External Knowledge:
    - Use special tags ?
- Framed as a [Contrastive Learning problem][contrastive].
- Experiment with [GPT3][gpt].
- Understand natural language better (why not ?!).
- Try dropping custom tokenization.
- Reinforcement learning for richer representation.
- Check out [Foundation models][foundation_models].
- Evaluate every dataset on [GPT][gpt].
- Check out [Structured State Space Sequence Models][S4] that can model **very long range dependencies**.
- Read the [Sentiment Neuron][sentiment-neuron] paper.
- Query [ChatGPT][chatgpt] and use its responses as context.
- Look at NeurIPS [Deep Bidirectional Language-Knowledge Graph Pretraining][dragon]
  - They improve pre-training using Knowledge Graphs.
- **Please read:** A [Frustratingly Easy Approach for Entity and Relation Extraction][pipeline]
- Current NER cost function doesn't take into account **how close your predicted span was** to the actual span.
- Use convolutional neural networks to improve span representations.
- Implement [SpanRel][spanrel]

# Experiments:
- Classify Span Representations after enriching tokens with external knowledge.
- Use names from WikiPedia.

[pipeline]: https://arxiv.org/pdf/2010.12812.pdf
[dragon]: https://arxiv.org/abs/2210.09338
[chatgpt]: https://openai.com/blog/chatgpt/
[S4]: https://arxiv.org/pdf/2111.00396.pdf
[foundation_models]: https://crfm.stanford.edu/2021/08/26/mistral.html
[gpt]: https://platform.openai.com/docs/introduction/overview?submissionGuid=9d64e167-19b7-4b8f-93f3-0cfec1d3b580
[contrastive]: https://arxiv.org/pdf/2208.14565v1.pdf
[transformer_ext]: https://arxiv.org/abs/2209.03528
[flert]: https://arxiv.org/abs/2011.06993
[mi-rim]: https://ceur-ws.org/Vol-3202/livingner-paper7.pdf
[spanrel]: https://aclanthology.org/2020.acl-main.192/
[sentiment-neuron]: https://arxiv.org/abs/1704.01444