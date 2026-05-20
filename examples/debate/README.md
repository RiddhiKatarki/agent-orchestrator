# Example: Multi-Agent Debate

Two debater agents argue opposite sides of a proposition across multiple rounds, then a judge agent delivers a scored verdict.

## Pipeline

```
Input: "AI will eliminate more jobs than it creates"
       │
       ▼ (parallel — opening statements)
  ┌────────────┐  ┌────────────┐
  │ Proponent  │  │  Opponent  │  Argue FOR and AGAINST simultaneously
  └────────────┘  └────────────┘
       │                │
       └────────────────┘
                │
                ▼ (loop — rebuttal rounds)
  ┌─────────────────────────────────┐
  │  ProponentRebuttal              │  Sequential within each round
  │  OpponentRebuttal               │  Reads opponent's latest argument
  └─────────────────────────────────┘
       (repeats DEBATE_ROUNDS times)
                │
                ▼
          ┌─────────┐
          │  Judge  │  Reads all arguments, scores both sides, picks winner
          └─────────┘
```

## Agents

| Agent | Reads | Writes |
|---|---|---|
| Proponent | `topic` | `for_argument` |
| Opponent | `topic` | `against_argument` |
| ProponentRebuttal | `topic`, `against_argument`, `against_rebuttal`, `_loop_iteration` | `for_rebuttal` |
| OpponentRebuttal | `topic`, `for_argument`, `for_rebuttal`, `_loop_iteration` | `against_rebuttal` |
| Judge | `topic`, `for_argument`, `against_argument`, `for_rebuttal`, `against_rebuttal` | `verdict` |

## Usage

```bash
# Default topic
python -m orchestrator.examples.debate.main

# Custom topic
python -m orchestrator.examples.debate.main "Remote work is better than office work"
python -m orchestrator.examples.debate.main "Nuclear energy should be expanded globally"
```

## Sample output

```
Proposition: "AI will eliminate more jobs than it creates"

Winner:  AGAINST
Scores:  FOR 6/10  vs  AGAINST 8/10

Reasoning: While the FOR side made compelling points about automation
displacing routine jobs, the AGAINST side provided stronger historical
evidence that technological revolutions consistently create more jobs
than they destroy...

Strongest argument: Historical precedent — every major technology wave
(steam, electricity, computers) ultimately expanded employment...

Weakest argument: The claim that AI is "fundamentally different" was
asserted but not sufficiently evidenced...
```

## Configuration

```bash
export DEBATE_ROUNDS=2  # number of rebuttal rounds (default: 2)
```
