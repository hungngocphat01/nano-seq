A minimal text sequence modeling library. Aims to create a neural machine translation model from scratch.  
Currently has a working Transformer implementation and a simple sentence classification trainer.  
Other details TBA.

- [x] Transformer implementation
- [x] Raw text data loading/collating logic
- [x] Proxy task to evaluate the implementation (sentence classification)
- [x] Common logic between different tasks (training loop, config management, etc.)
- [ ] NMT
  - [ ] Parallel (bilingual) dataset implementation
  - [ ] Greedy decoding
  - [ ] Beam search
- [ ] NMT++
  - [ ] Memory optimization (lazy-loading dataset)
  - [ ] KV caching
