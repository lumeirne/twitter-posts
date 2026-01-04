import dspy
import os

# --- Configure LLM ---
# For openai models
# lm = dspy.LM("openai/gpt-4o-mini")
# Set up dspy to use OpenRouter.
# It reads the API key from the OPENROUTER_API_KEY environment variable.
llm = dspy.LM(
    model="openrouter/kwaipilot/kat-coder-pro:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
dspy.settings.configure(lm=llm)
# --- End LLM Configuration ---


# 2. Define Signatures (WHAT you want)
class ExtractClaims(dspy.Signature):
    """Extract main claims from research text"""
    paper_text: str = dspy.InputField(desc="Research paper content")
    claims: list[str] = dspy.OutputField(desc="Key claims made")


class AssessNovelty(dspy.Signature):
    """Assess if claims are novel"""
    claims: list[str] = dspy.InputField()
    novelty_score: int = dspy.OutputField(desc="Score from 1-10")
    reasoning: str = dspy.OutputField()


class GenerateSummary(dspy.Signature):
    """Generate final summary"""
    claims: list[str] = dspy.InputField()
    novelty_score: int = dspy.InputField()
    summary: str = dspy.OutputField(desc="2-3 sentence summary")


# 3. Build Pipeline as a Module
class PaperAnalyzer(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought(ExtractClaims)
        self.assess = dspy.ChainOfThought(AssessNovelty)
        self.summarize = dspy.ChainOfThought(GenerateSummary)
    
    def forward(self, paper_text: str):
        # Step 1: Extract claims
        claims_result = self.extract(paper_text=paper_text)
        
        # Step 2: Assess novelty
        novelty_result = self.assess(claims=claims_result.claims)
        
        # Step 3: Generate summary
        summary_result = self.summarize(
            claims=claims_result.claims,
            novelty_score=novelty_result.novelty_score
        )
        
        return dspy.Prediction(
            claims=claims_result.claims,
            novelty_score=novelty_result.novelty_score,
            reasoning=novelty_result.reasoning,
            summary=summary_result.summary
        )


# 4. Use it!
analyzer = PaperAnalyzer()

paper = """
We introduce a new architecture called Transformer that relies 
entirely on attention mechanisms, dispensing with recurrence 
and convolutions. Experiments show the model achieves 28.4 BLEU 
on English-to-German translation, surpassing existing best results.
"""

result = analyzer(paper_text=paper)

print(f"Claims: {result.claims}")
print(f"Novelty: {result.novelty_score}/10")
print(f"Summary: {result.summary}")