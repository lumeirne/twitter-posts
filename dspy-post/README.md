# DSPy Paper Analyzer

This project demonstrates how to use the `dspy` library to create a simple pipeline for analyzing research paper text. The pipeline extracts key claims, assesses their novelty, and generates a concise summary.

## How it Works

The project is built using `dspy`, a framework for programming with language models. It defines a series of "signatures" that specify the desired input and output of a language model task:

- `ExtractClaims`: Takes paper text and returns a list of claims.
- `AssessNovelty`: Takes a list of claims and returns a novelty score (1-10) and reasoning.
- `GenerateSummary`: Takes the claims and novelty score to produce a final summary.

These signatures are chained together in a `dspy.Module` called `PaperAnalyzer` to form a complete pipeline.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your environment:** This project requires an API key from OpenRouter.
   - I have used the openrouter free models, you can use your own.
   - Get your API key from [https://openrouter.ai/](https://openrouter.ai/).
   - Set it as an environment variable:
     ```bash
     export OPENROUTER_API_KEY="your-api-key-here"
     ```

## Usage

Run the script:
```bash
python app.py
```

## Example Output

The script includes an example using a short text about the Transformer architecture. The output will look something like this:

```
Claims: ['A new architecture called Transformer is introduced.', 'The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions.', 'Experiments show the model achieves 28.4 BLEU on English-to-German translation, surpassing existing best results.']
Novelty: 8/10
Summary: The paper introduces a novel Transformer architecture that relies solely on attention mechanisms, outperforming existing models on English-to-German translation tasks. This new approach, which dispenses with recurrence and convolutions, could represent a significant advancement in the field of machine translation.
```
