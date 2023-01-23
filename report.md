### Models
- Bert Linear Sequence Labeling
    - Bert Layer representing tokens.
    - Linear Classifier Layer classifying each token to BIO labels.
- Bert Seq Label CRF
    - Bert Layer representing tokens.
    - A CRF Classifier Layer classiying each token to BIO labels. 
- [SpanNER](https://arxiv.org/abs/2106.00641)
    - Bert Layer representing tokens.
    - Layer that represents all possible spans using the beginning and ending token representations.
    - Linear Classifier Layer classifying each token to BIO labels.

### Errors
#### Multiconer Fine
- Cannot tell difference between PublicCorp and PrivateCorp.
- Definitely needs external knowledge.
- False Positives:
  > the adoption of the ( of charles v ) in 1532 made **inquisitional** procedures empirical law.  
  
  lacks syntax knowledge.
  > **it** was written by tom johnston .

  Respectable mistake.
