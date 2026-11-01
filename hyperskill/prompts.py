"""
System prompts for HyperSkill agents across all modalities and hop types.
"""

# ---------------------------------------------------------------------------
# Base reasoning prompt (Text / VL reasoning tasks — no search tools)
# ---------------------------------------------------------------------------

REASONING_PROMPT = """\
Answer the following question. Output ONLY your final answer inside <answer>...</answer> tags. Do NOT show your work.
"""

# ---------------------------------------------------------------------------
# Search-augmented prompt (Text / VL websearch tasks)
# ---------------------------------------------------------------------------

SEARCH_AGENT_PROMPT = """\
You are an expert research assistant equipped with search tools.
Analyze the question and decide whether to search for more information or to
provide a final answer directly.
{tool_desc}

You must respond in ONE of the following formats:

  Format A — Direct answer (when confident):
    <reason>...</reason>
    <answer>...</answer>

  Format B — Search for more information:
    <reason>...</reason>
    <TOOL_TAG>specific query</TOOL_TAG>

Replace TOOL_TAG with one of the available tools listed above.
"""

# ---------------------------------------------------------------------------
# Omni-modal agent prompt (Omni tasks — text + image + audio + video)
# ---------------------------------------------------------------------------

OMNI_AGENT_PROMPT = """\
You are an expert multimodal reasoning assistant. You can process text,
images, audio transcripts, and video descriptions. Analyze all provided
modalities together to derive a precise answer.
{tool_desc}

You must respond in ONE of the following formats:

  Format A — Direct answer (when confident):
    <reason>...</reason>
    <answer>...</answer>

  Format B — Search for more information:
    <reason>...</reason>
    <TOOL_TAG>specific query</TOOL_TAG>

Replace TOOL_TAG with one of the available tools listed above.
"""

# ---------------------------------------------------------------------------
# After-search continuation prompt
# ---------------------------------------------------------------------------

AFTER_SEARCH_PROMPT = """\
Based on the search results provided, continue your reasoning and either
search again (if needed) or provide the final answer.

Original question: {question}

{search_results}

Now provide your next step:
"""

# ---------------------------------------------------------------------------
# Contrastive step-level skill distillation + counterfactual verification
# ---------------------------------------------------------------------------
SKILL_CONTRASTIVE_EXTRACT_SP = """\
You are an expert at analyzing reasoning trajectories to extract reusable skills.

Your task: given a question with GOOD attempts (high reward) and BAD attempts (low reward), perform **step-level contrastive analysis** to find exactly WHERE and WHY the good attempts succeeded.

<Instructions>
1. **Decompose** each attempt into numbered reasoning steps.
2. **Align** steps across good and bad attempts — find corresponding steps that address the same sub-problem.
3. **Identify divergence points** — the specific steps where good and bad attempts DIFFER in strategy or correctness.
4. **Extract atomic skills** — from each divergence point, extract the generalizable skill that the good attempt used (NOT specific to this question).
5. **Identify skill recipes** — if multiple atomic skills worked TOGETHER in successful attempts but were missing together in failures, group them as a recipe.
</Instructions>

<Output Format>
<Step Alignment>
[Show the aligned steps of good vs bad attempts, marking divergence points with ⚡]
</Step Alignment>

<Divergence Analysis>
[For each divergence point: what the good attempt did vs what the bad attempt did, and WHY the good approach worked]
</Divergence Analysis>

<Atomic Skills>
[One per line, extracted from divergence points:]
SKILL: <name (3-7 words)> | <what to do and when (1-2 sentences)>
</Atomic Skills>

<Skill Recipes>
[Combinations of atomic skills that must be used TOGETHER for success:]
RECIPE: <recipe name> | <skill_1 + skill_2 + ...> | <when to apply this combination>
</Skill Recipes>
</Output Format>
"""

SKILL_CONTRASTIVE_EXTRACT_UP = """\
<Question>
{question}
</Question>

<Gold Answer>
{gold_answer}
</Gold Answer>

<Good Attempts>
{good_attempts}
</Good Attempts>

<Bad Attempts>
{bad_attempts}
</Bad Attempts>

Perform step-level contrastive analysis as instructed.
"""

# Counterfactual verification: would this skill help a failed trajectory?
SKILL_COUNTERFACTUAL_VERIFY_SP = """\
You are evaluating whether a proposed reasoning skill would help an agent that FAILED a question.

Given:
- A question the agent got WRONG
- The agent's FAILED reasoning trajectory
- A proposed SKILL

Judge: If the agent had known and applied this skill, would it likely have answered correctly?

Output EXACTLY one line:
VERDICT: YES | <confidence 0-1> | <brief reason>
or
VERDICT: NO | <confidence 0-1> | <brief reason>
"""

SKILL_COUNTERFACTUAL_VERIFY_UP = """\
<Question>
{question}
</Question>

<Gold Answer>
{gold_answer}
</Gold Answer>

<Failed Trajectory>
{failed_trajectory}
</Failed Trajectory>

<Proposed Skill>
{skill}
</Proposed Skill>

Would this skill have helped? Output VERDICT line only.
"""

# ---------------------------------------------------------------------------
# Skill-augmented reasoning prompt (inject retrieved skills into context)
# ---------------------------------------------------------------------------

SKILL_AUGMENTED_PROMPT = """\
Tip from prior experience: {skills_context}
"""

# P7 -- Strategy Hint Synthesis (paper App. A.3, verbatim)
SKILL_SUMMARY_PROMPT = """\
Given a question and a set of retrieved reasoning skills, write a concise
strategy hint (2-3 sentences) that synthesizes these skills specifically
for this question. Focus on HOW to combine them, not just listing them.
Question: {question}
Retrieved Skills:
{skills_raw}
Strategy hint (2-3 sentences, actionable):"""

# Injection variant (ii), schema-constrained hint: output-schema and tool-budget
# constraints appended to P7 (schema_constrained_hint=True)
SKILL_SUMMARY_SCHEMA = """
Constraints for the hint: it must not change the required output format (the agent ends with a short final answer inside <answer>...</answer>, as short as the question allows, never a full sentence when a name, number or phrase suffices).{tool_clause}"""
SKILL_SUMMARY_SCHEMA_TOOLS = " It must not ask for more than {limit} tool calls in total, and the agent must always finish with a final <answer> even if a search did not find everything."

# Injection variant (iii), tool-neutral hint (tool_neutral_hint=True, tool regimes)
SKILL_SUMMARY_TOOL_NEUTRAL = """
The hint must not tell the agent to search, look anything up or use any tool; focus only on how to reason about the given question and input and how to phrase the final answer."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_skills_context(skills: list) -> str:
    """Format retrieved skills for prompt injection."""
    if not skills:
        return "(No relevant skills found)"

    # HyperSkill format (terse)
    lines = []
    for skill in skills:
        lines.append(f"{skill.name}: {skill.description}")
    return " | ".join(lines)


def build_initial_prompt(question: str, modality: str, tool_desc: str = "") -> str:
    """Build the initial prompt string for a given question and modality."""
    if modality == "omni":
        base = OMNI_AGENT_PROMPT.format(tool_desc=tool_desc)
    elif tool_desc:  # has tools → use search agent
        base = SEARCH_AGENT_PROMPT.format(tool_desc=tool_desc)
    else:
        base = REASONING_PROMPT
    return base + f"\n\nHere is the question:\n{question}"


def build_skill_augmented_prompt(
    question: str,
    modality: str,
    skills: list,
    tool_desc: str = "",
    skills_context: str = None,
) -> str:
    """Build a skill-augmented initial prompt.

    When NO skill passes the retrieval gates, fall back to the plain
    initial prompt — no
    "(No relevant skills found)" banner is injected, since it adds
    irrelevant tokens that can perturb the model.
    """
    base = build_initial_prompt(question, modality, tool_desc=tool_desc)
    if skills_context:  # pre-formatted context (P7 hint and/or answer-form line)
        return SKILL_AUGMENTED_PROMPT.format(skills_context=skills_context) + "\n\n" + base
    if not skills:
        return base
    skill_ctx = format_skills_context(skills)
    skill_header = SKILL_AUGMENTED_PROMPT.format(skills_context=skill_ctx)
    return skill_header + "\n\n" + base
