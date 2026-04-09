# dann-domain-adaptation

Domain-Adversarial Neural Network (DANN) for site-invariant feature learning in computational pathology. Gradient Reversal Layer implementation with progressive lambda schedule.

## Method
DANN (Ganin et al., 2016) adds a domain classifier branch to a feature extractor with a Gradient Reversal Layer (GRL). During backprop, gradients from the domain branch are reversed, forcing the encoder to learn features that are indistinguishable across sites/scanners while remaining predictive for the task.

Lambda schedule: λ(p) = 2/(1 + exp(−γp)) − 1, where p is training progress [0,1].

## Installation
```bash
pip install dann-domain-adaptation
```

## Quick start
```python
from dann_domain_adaptation import DANNModel, GradientReversalLayer

model = DANNModel(n_classes=2, n_domains=5)
# forward pass returns task_logits, domain_logits
task_logits, domain_logits = model(x, lambda_=0.5)
```

## License
MIT
