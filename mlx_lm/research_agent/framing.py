"""Topic framing — classify topic type and build dimension map."""

from .memory import DIMENSION_TEMPLATES
from ._utils import call_cheap, call_model


KNOWN_TYPES = set(DIMENSION_TEMPLATES.keys())


def infer_topic_type(topic: str, model, tokenizer, args,
                     chat_template_kwargs=None) -> str:
    """Use the model to classify the topic type."""
    messages = [
        {"role": "system", "content": f"Classify this topic into exactly one type: {', '.join(sorted(KNOWN_TYPES))}. Output only the type name."},
        {"role": "user", "content": topic},
    ]
    result = call_cheap(messages, max_tokens=16, temp=0.0)
    if not result:
        result = call_model(messages, max_tokens=16, model=model,
                            tokenizer=tokenizer, args=args,
                            chat_template_kwargs=chat_template_kwargs,
                            temp=0.0)
    result = result.strip().lower().rstrip(".")
    if result in KNOWN_TYPES:
        return result
    return "general_topic"


def build_dimension_map(topic_type: str, topic: str = "") -> dict:
    """Build dimension map for a topic.

    If a topic string is provided, tries the small model first to generate
    custom dimensions tailored to the topic (e.g. "knee pain" → causes,
    symptoms, treatments). Falls back to the static template if the small
    model is unavailable or parsing fails.
    """
    if topic:
        result = call_cheap(
            [
                {"role": "system", "content": f"""Given the research topic, generate 6-10 research dimensions to investigate. Each dimension should be a single lowercase word or underscore_separated phrase. Output one per line, no numbering, no explanation.

Example:
Topic: knee pain
causes
symptoms
diagnosis
treatment_options
home_remedies
prevention
when_to_see_a_doctor
recovery_time

Topic: super bowl 2025
teams
score
halftime_show
mvp
key_plays
commercials
attendance

Topic: {topic}"""},
                {"role": "user", "content": topic},
            ],
            max_tokens=128,
            temp=0.3,
        )
        if result:
            lines = [l.strip().lower().replace(" ", "_") for l in result.splitlines()
                     if l.strip() and not l.strip().startswith("-")]
            lines = [l for l in lines if l[0].isalpha() and len(l) > 2][:12]
            if len(lines) >= 4:
                return {d: 0.0 for d in lines}

    # Fallback to static template
    dims = DIMENSION_TEMPLATES.get(topic_type, DIMENSION_TEMPLATES["general_topic"])
    return {d: 0.0 for d in dims}
