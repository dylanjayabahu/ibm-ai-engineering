# Fine-Tuning vs RAG

## Fine-Tuning
- Adapt a **pre-trained model** to a **specific domain/task** by updating weights.
- Uses labeled domain data for training.
- **InstructLab** enables local fine-tuning and structured knowledge updates via taxonomy.
- **Pros:** domain expertise, high accuracy in narrow tasks.
- **Cons:** expensive to train, hard to update (retrain needed), static knowledge.

## RAG (Retrieval-Augmented Generation)
- Combines **retrieval + generation**.
- Retrieves relevant documents from a **knowledge base** during query time.
- **Pros:** dynamic knowledge updates, less training, handles broad info.
- **Cons:** depends on retrieval quality, external DB required.

## Key Differences
| Aspect | Fine-Tuning | RAG |
|--------|-------------|-----|
| Training | Heavy | Minimal |
| Adaptability | Specialized | Flexible |
| Updating Knowledge | Retrain needed | Update DB only |
| Resources | High compute | Lower compute |
| Accuracy | High in-domain | High w/ good retrieval |
| Best Use | Static, narrow domains | Dynamic, broad info |

## When to Use
- **Use Fine-Tuning** → stable domain + specialized knowledge.
- **Use RAG** → evolving information + broad coverage.

