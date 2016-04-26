# Knowledge Graph Paper Notes

## Overall


## Notes

### Embedding entities and relations for learning and inference in knowledge bases

bishan, wen-tau, xiaodong, jianfeng, lideng, cornell and msr, ICLR15  

Problem:   

1. Lack of *careful comparison* with TranE/NTN/RESCAL on design choices that affect the learning results.
2. Evaluated on link prediction task, hard to explain *what relational properties are being captured* and to what extent they are *captured* during the embedding process.

Contributions:  

1. Present a general framework for multirelational learning that unifies most multi-relational embedding models developed in the past, including NTN, TranE.
2. Empirically evaluate different choices of entity representations and relation representations under this framework on the canonical link predictin task and show that a simple bilinear formulation achieves new state-of-the-art results for the task.
3. Propse and evaluate a novel approach that utilized the learned embeddings to mine logical Horn-clause. Demonstrate that out blabla outperforms a SOTA rule mining system AMIE on mining rules that involes compositional reasoning.

Content:  

NTN/TransE differ in different parametrization of relation operators. 
