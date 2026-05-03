"""
Vera AI Composer — LLM-powered message composition engine.
Uses Groq API with Llama models for fast, high-quality message generation.

Architecture:
  - Dispatch by trigger.kind → specialized prompt variant
  - Build structured prompt from 4 contexts (category, merchant, trigger, customer?)
  - Post-validate output for CTA shape, language, taboos
"""

import os
import json
import re
from typing import Optional
from groq import Groq

# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# System Prompt — the master composer instruction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages for merchants and their customers.

CRITICAL RULES:
1. NEVER fabricate data — only use facts from the provided contexts
2. Keep messages concise — WhatsApp-appropriate, not email-length
3. Match the category voice exactly (clinical-peer for dentists, warm for salons, operator-to-operator for restaurants, coaching for gyms, trustworthy for pharmacies)
4. Hindi-English code-mix is encouraged when the merchant's languages include "hi"
5. NEVER use taboo vocabulary listed in the category voice
6. Anchor on verifiable facts: numbers, dates, sources, prices
7. Single primary CTA — binary (YES/STOP) for action triggers, open-ended for info triggers
8. No long preambles. No "I hope you're doing well". Get to the point.
9. Service+price ("Dental Cleaning @ ₹299") beats generic discounts ("10% off")
10. Use one or more compulsion levers: specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity, asking the merchant, single binary commitment.

RESPOND ONLY WITH VALID JSON matching this schema:
{
  "body": "the WhatsApp message text",
  "cta": "binary_yes_no | open_ended | none | multi_choice_slot",
  "send_as": "vera | merchant_on_behalf",
  "suppression_key": "from the trigger",
  "rationale": "1-2 sentence explanation of why this message, what compulsion levers used"
}"""

# ---------------------------------------------------------------------------
# Trigger-kind specific prompt builders
# ---------------------------------------------------------------------------


def _build_research_digest_prompt(category: dict, merchant: dict, trigger: dict) -> str:
    """Research/compliance/CDE digest framing."""
    digest_items = category.get("digest", [])
    top_item_id = trigger.get("payload", {}).get("top_item_id", "")
    top_item = next((d for d in digest_items if d.get("id") == top_item_id), None)

    digest_text = json.dumps(top_item, indent=2) if top_item else json.dumps(digest_items[:2], indent=2)

    return f"""COMPOSE a research-digest message for this merchant.

TRIGGER KIND: {trigger.get('kind', 'research_digest')}
This is an EXTERNAL research/compliance update. Frame it as sharing useful professional knowledge — peer-to-peer, not promotional.

KEY DIGEST ITEM:
{digest_text}

FRAMING GUIDANCE:
- Lead with the source and key finding
- Connect it to this merchant's specific situation (patient cohort, case-mix, signals)
- Offer to do something useful (draft patient-ed post, pull the abstract, schedule a post)
- Source citation at the end
- CTA: open_ended (let them ask for more)"""


def _build_perf_trigger_prompt(category: dict, merchant: dict, trigger: dict) -> str:
    """Performance dip/spike framing."""
    payload = trigger.get("payload", {})
    kind = trigger.get("kind", "")
    metric = payload.get("metric", "views")
    delta = payload.get("delta_pct", 0)
    direction = "up" if delta > 0 else "down"
    abs_delta = abs(int(delta * 100))

    return f"""COMPOSE a performance-{direction} message for this merchant.

TRIGGER KIND: {kind}
The merchant's {metric} went {direction} {abs_delta}% in the last {payload.get('window', '7d')}.

PAYLOAD: {json.dumps(payload)}

FRAMING GUIDANCE for SPIKE (up):
- Celebrate the win with specific numbers
- Attribute it to something they did (recent post, offer, season)
- Suggest next action to sustain momentum
- CTA: open_ended

FRAMING GUIDANCE for DIP (down):
- Don't alarm — normalize if seasonal, otherwise frame as recoverable
- Use peer comparison if relevant (their CTR vs peer median)
- Offer a specific fix ("Want me to draft 3 fresh posts?")
- Use loss-aversion framing ("You're missing X searches")
- CTA: binary_yes_no"""


def _build_recall_due_prompt(category: dict, merchant: dict, trigger: dict, customer: dict) -> str:
    """Customer recall/appointment reminder framing."""
    payload = trigger.get("payload", {})
    slots = payload.get("available_slots", [])
    slot_text = " ya ".join([s.get("label", "") for s in slots[:3]]) if slots else "this week"

    return f"""COMPOSE a customer-facing recall reminder sent from the MERCHANT's WhatsApp number.

TRIGGER KIND: recall_due (customer-scoped)
This message is sent as the MERCHANT (send_as = "merchant_on_behalf"), NOT as Vera.

CUSTOMER: {json.dumps(customer.get('identity', {}))}
RELATIONSHIP: {json.dumps(customer.get('relationship', {}))}
STATE: {customer.get('state', 'unknown')}
PREFERENCES: {json.dumps(customer.get('preferences', {}))}

RECALL DETAILS:
- Service due: {payload.get('service_due', 'regular checkup')}
- Last service: {payload.get('last_service_date', 'unknown')}
- Available slots: {slot_text}

FRAMING GUIDANCE:
- Address customer by name
- Mention the merchant's name/clinic name
- Reference time since last visit
- Offer specific slots matching customer's preference
- Include the service price from merchant's active offers
- Match customer's language_pref
- CTA: multi_choice_slot (for booking) or binary_yes_no
- send_as MUST be "merchant_on_behalf"
- Keep warm but clinical — no overclaims"""


def _build_festival_prompt(category: dict, merchant: dict, trigger: dict) -> str:
    """Festival/seasonal opportunity framing."""
    payload = trigger.get("payload", {})

    return f"""COMPOSE a festival/seasonal opportunity message for this merchant.

TRIGGER KIND: {trigger.get('kind', 'festival_upcoming')}
FESTIVAL/EVENT: {payload.get('festival', payload.get('season', 'upcoming event'))}
DAYS UNTIL: {payload.get('days_until', '?')}

PAYLOAD: {json.dumps(payload)}

FRAMING GUIDANCE:
- Connect the festival/season to their specific category and business
- Suggest a concrete action (run a themed offer, update GBP post, stock up)
- Reference their current offers and performance
- If far away (>30 days), frame as early-mover advantage
- CTA: binary_yes_no or open_ended"""


def _build_competitor_prompt(category: dict, merchant: dict, trigger: dict) -> str:
    """Competitor opened nearby framing."""
    payload = trigger.get("payload", {})

    return f"""COMPOSE a competitor-awareness message for this merchant.

TRIGGER KIND: competitor_opened
A new competitor opened nearby.

COMPETITOR: {payload.get('competitor_name', 'New competitor')}
DISTANCE: {payload.get('distance_km', '?')} km away
THEIR OFFER: {payload.get('their_offer', 'unknown')}
OPENED: {payload.get('opened_date', 'recently')}

FRAMING GUIDANCE:
- Use curiosity framing ("noticed a new listing near you")
- Don't be alarmist — frame as market intelligence
- Compare their offer vs merchant's current offer if possible
- Suggest a defensive action (update profile, refresh offer, add photos)
- Social proof: mention merchant's existing strengths (rating, reviews)
- CTA: open_ended (let merchant decide what to do)"""


def _build_renewal_prompt(category: dict, merchant: dict, trigger: dict) -> str:
    """Subscription renewal framing."""
    payload = trigger.get("payload", {})
    perf = merchant.get("performance", {})

    return f"""COMPOSE a subscription renewal message for this merchant.

TRIGGER KIND: renewal_due
DAYS REMAINING: {payload.get('days_remaining', merchant.get('subscription', {}).get('days_remaining', '?'))}
PLAN: {payload.get('plan', merchant.get('subscription', {}).get('plan', '?'))}
RENEWAL AMOUNT: ₹{payload.get('renewal_amount', '?')}

MERCHANT PERFORMANCE (what they'd lose):
- Views: {perf.get('views', '?')} in last 30d
- Calls: {perf.get('calls', '?')}
- Directions: {perf.get('directions', '?')}

FRAMING GUIDANCE:
- Lead with value delivered (their performance numbers)
- Frame as what they'd lose, not what they'd gain (loss aversion)
- Keep it factual, not salesy
- Mention specific numbers from their performance
- CTA: binary_yes_no"""


def _build_review_theme_prompt(category: dict, merchant: dict, trigger: dict) -> str:
    """Review theme emerged framing."""
    payload = trigger.get("payload", {})

    return f"""COMPOSE a review-theme alert message for this merchant.

TRIGGER KIND: review_theme_emerged
THEME: {payload.get('theme', 'unknown')}
OCCURRENCES (30d): {payload.get('occurrences_30d', '?')}
TREND: {payload.get('trend', 'stable')}
SAMPLE QUOTE: "{payload.get('common_quote', '')}"

FRAMING GUIDANCE:
- Share the pattern factually with the specific count
- Include the customer quote if available
- Frame as actionable intelligence, not criticism
- Suggest a concrete response (reply to reviews, update hours, add a note to GBP)
- Reciprocity: "I spotted this pattern, thought you'd want to know"
- CTA: open_ended"""


def _build_customer_lapsed_prompt(category: dict, merchant: dict, trigger: dict, customer: dict) -> str:
    """Customer lapsed/winback framing."""
    payload = trigger.get("payload", {})

    return f"""COMPOSE a customer win-back message sent from the MERCHANT's WhatsApp number.

TRIGGER KIND: {trigger.get('kind', 'customer_lapsed')} (customer-scoped)
send_as MUST be "merchant_on_behalf"

CUSTOMER: {json.dumps(customer.get('identity', {}))}
RELATIONSHIP: {json.dumps(customer.get('relationship', {}))}
DAYS SINCE LAST VISIT: {payload.get('days_since_last_visit', '?')}
PREVIOUS FOCUS: {payload.get('previous_focus', payload.get('previous_membership_months', 'unknown'))}

FRAMING GUIDANCE:
- Warm, personal tone — they're a returning customer, not a new lead
- Reference their history (services, visits, focus area)
- Offer a specific incentive from the merchant's active offers
- Low-friction CTA
- Match customer's language_pref
- send_as = "merchant_on_behalf" """


def _build_planning_intent_prompt(category: dict, merchant: dict, trigger: dict) -> str:
    """Active planning intent framing — merchant asked for help planning something."""
    payload = trigger.get("payload", {})

    return f"""COMPOSE a planning-response message for this merchant.

TRIGGER KIND: active_planning_intent
The merchant expressed interest in planning something specific.

INTENT TOPIC: {payload.get('intent_topic', 'unknown')}
MERCHANT'S LAST MESSAGE: "{payload.get('merchant_last_message', '')}"

FRAMING GUIDANCE:
- This merchant is ENGAGED — they asked for help. Honor that.
- Provide a concrete, actionable proposal (not more questions)
- Include specific numbers: pricing, timeline, format
- Reference category trends or peer data to support the proposal
- Offer to execute ("Want me to draft the post / create the package / set it up?")
- CTA: binary_yes_no (to approve the plan)"""


def _build_generic_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    """Fallback for any trigger kind not specifically handled."""
    payload = trigger.get("payload", {})
    kind = trigger.get("kind", "unknown")
    scope = trigger.get("scope", "merchant")

    customer_section = ""
    if customer:
        customer_section = f"""
CUSTOMER (this is a customer-scoped trigger — send_as = "merchant_on_behalf"):
{json.dumps(customer.get('identity', {}), indent=2)}
RELATIONSHIP: {json.dumps(customer.get('relationship', {}), indent=2)}
STATE: {customer.get('state', 'unknown')}"""

    return f"""COMPOSE a message for this {scope}-scoped trigger.

TRIGGER KIND: {kind}
SOURCE: {trigger.get('source', 'unknown')}
URGENCY: {trigger.get('urgency', 1)}/5

PAYLOAD: {json.dumps(payload, indent=2)}
{customer_section}

FRAMING GUIDANCE:
- Connect the trigger event to the merchant's specific situation
- Use at least one compulsion lever (curiosity, loss aversion, social proof, etc.)
- Be specific — anchor on verifiable facts from the contexts
- {"send_as = 'merchant_on_behalf'" if customer else "send_as = 'vera'"}
- CTA: binary_yes_no or open_ended as appropriate"""


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------

def _build_context_block(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    """Build the structured context block injected into every prompt."""
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    cust_agg = merchant.get("customer_aggregate", {})
    voice = category.get("voice", {})
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    conv_history = merchant.get("conversation_history", [])
    last_conv = conv_history[-1] if conv_history else None

    lines = [
        "=== CONTEXT ===",
        f"CATEGORY: {category.get('slug', 'unknown')}",
        f"VOICE: tone={voice.get('tone', '?')}, taboos={voice.get('vocab_taboo', [])}",
        f"PEER STATS: avg_rating={category.get('peer_stats', {}).get('avg_rating', '?')}, avg_ctr={category.get('peer_stats', {}).get('avg_ctr', '?')}, avg_views={category.get('peer_stats', {}).get('avg_views_30d', '?')}",
        "",
        f"MERCHANT: {identity.get('name', '?')}",
        f"  Owner: {identity.get('owner_first_name', '?')}",
        f"  Location: {identity.get('locality', '?')}, {identity.get('city', '?')}",
        f"  Languages: {identity.get('languages', ['en'])}",
        f"  Verified: {identity.get('verified', False)}",
        f"  Subscription: {sub.get('status', '?')} ({sub.get('plan', '?')}), {sub.get('days_remaining', '?')} days remaining",
        f"  Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, directions={perf.get('directions', '?')}, ctr={perf.get('ctr', '?')}",
        f"  7d delta: views {perf.get('delta_7d', {}).get('views_pct', '?')}, calls {perf.get('delta_7d', {}).get('calls_pct', '?')}",
        f"  Active offers: {[o.get('title') for o in active_offers] if active_offers else 'NONE'}",
        f"  Signals: {merchant.get('signals', [])}",
        f"  Customer aggregate: {json.dumps(cust_agg)}",
        f"  Review themes: {json.dumps(merchant.get('review_themes', []))}",
    ]

    if last_conv:
        lines.append(f"  Last Vera interaction: {last_conv.get('ts', '?')} — \"{last_conv.get('body', '')[:100]}...\" ({last_conv.get('engagement', '?')})")

    lines.append("")
    lines.append(f"TRIGGER: kind={trigger.get('kind', '?')}, source={trigger.get('source', '?')}, urgency={trigger.get('urgency', '?')}/5")
    lines.append(f"  suppression_key: {trigger.get('suppression_key', '')}")

    if customer:
        lines.append("")
        lines.append(f"CUSTOMER: {customer.get('identity', {}).get('name', '?')}")
        lines.append(f"  Language pref: {customer.get('identity', {}).get('language_pref', 'english')}")
        lines.append(f"  State: {customer.get('state', '?')}")
        lines.append(f"  Visits: {customer.get('relationship', {}).get('visits_total', '?')}")
        lines.append(f"  Last visit: {customer.get('relationship', {}).get('last_visit', '?')}")
        lines.append(f"  Services: {customer.get('relationship', {}).get('services_received', [])}")
        lines.append(f"  Preferences: {json.dumps(customer.get('preferences', {}))}")

    return "\n".join(lines)


def _dispatch_prompt(kind: str, category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    """Route to the right prompt builder based on trigger kind."""
    if kind in ("research_digest", "regulation_change", "cde_opportunity"):
        return _build_research_digest_prompt(category, merchant, trigger)
    elif kind in ("perf_dip", "perf_spike", "seasonal_perf_dip"):
        return _build_perf_trigger_prompt(category, merchant, trigger)
    elif kind == "recall_due" and customer:
        return _build_recall_due_prompt(category, merchant, trigger, customer)
    elif kind in ("festival_upcoming", "ipl_match_today", "category_seasonal"):
        return _build_festival_prompt(category, merchant, trigger)
    elif kind == "competitor_opened":
        return _build_competitor_prompt(category, merchant, trigger)
    elif kind in ("renewal_due", "winback_eligible"):
        return _build_renewal_prompt(category, merchant, trigger)
    elif kind == "review_theme_emerged":
        return _build_review_theme_prompt(category, merchant, trigger)
    elif kind in ("customer_lapsed_soft", "customer_lapsed_hard") and customer:
        return _build_customer_lapsed_prompt(category, merchant, trigger, customer)
    elif kind == "active_planning_intent":
        return _build_planning_intent_prompt(category, merchant, trigger)
    elif kind in ("trial_followup", "chronic_refill_due", "appointment_tomorrow") and customer:
        return _build_customer_lapsed_prompt(category, merchant, trigger, customer)
    else:
        return _build_generic_prompt(category, merchant, trigger, customer)


def compose(category: dict, merchant: dict, trigger: dict,
            customer: Optional[dict] = None) -> dict:
    """
    Main composition function. Takes 4 contexts → returns composed message dict.

    Returns:
        dict with keys: body, cta, send_as, suppression_key, rationale
    """
    kind = trigger.get("kind", "unknown")

    # Build the full prompt
    context_block = _build_context_block(category, merchant, trigger, customer)
    kind_prompt = _dispatch_prompt(kind, category, merchant, trigger, customer)

    full_prompt = f"""{context_block}

{kind_prompt}

Remember: respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

    # Call LLM
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback if LLM fails
        return _fallback_compose(category, merchant, trigger, customer, str(e))

    # Parse and validate
    return _parse_and_validate(raw, trigger, customer)


def _parse_and_validate(raw: str, trigger: dict, customer: Optional[dict]) -> dict:
    """Parse LLM output and validate/fix the response."""
    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        return _fallback_from_raw(raw, trigger, customer)

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError:
        return _fallback_from_raw(raw, trigger, customer)

    # Validate required fields
    required = ["body", "cta", "send_as", "suppression_key", "rationale"]
    for field in required:
        if field not in result or not result[field]:
            if field == "suppression_key":
                result["suppression_key"] = trigger.get("suppression_key", "")
            elif field == "send_as":
                result["send_as"] = "merchant_on_behalf" if customer else "vera"
            elif field == "cta":
                result["cta"] = "open_ended"
            elif field == "rationale":
                result["rationale"] = f"Composed for trigger kind={trigger.get('kind', '?')}"
            elif field == "body":
                return _fallback_compose({}, {}, trigger, customer, "Empty body from LLM")

    # Fix send_as for customer-scoped
    if customer and trigger.get("scope") == "customer":
        result["send_as"] = "merchant_on_behalf"

    # Ensure suppression_key matches trigger
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get("suppression_key", "")

    return result


def _fallback_from_raw(raw: str, trigger: dict, customer: Optional[dict]) -> dict:
    """If JSON parsing fails, use the raw text as body."""
    # Clean up raw text
    body = raw.strip()
    if body.startswith("```"):
        body = re.sub(r'^```\w*\n?', '', body)
        body = re.sub(r'\n?```$', '', body)

    return {
        "body": body[:500] if body else "Hi, Vera here. How can I help today?",
        "cta": "open_ended",
        "send_as": "merchant_on_behalf" if customer else "vera",
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": f"Fallback: LLM output wasn't valid JSON. Trigger kind={trigger.get('kind', '?')}",
    }


def _fallback_compose(category: dict, merchant: dict, trigger: dict,
                       customer: Optional[dict], error: str) -> dict:
    """Emergency fallback when LLM call fails entirely."""
    name = merchant.get("identity", {}).get("name", "there")
    return {
        "body": f"Hi {name}, Vera here — quick check-in. Want to review your latest performance numbers?",
        "cta": "open_ended",
        "send_as": "merchant_on_behalf" if customer else "vera",
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": f"Fallback due to LLM error: {error[:100]}",
    }


# ---------------------------------------------------------------------------
# Reply composer — for multi-turn conversations
# ---------------------------------------------------------------------------

def compose_reply(category: dict, merchant: dict, trigger: dict,
                  conversation_history: list, merchant_message: str,
                  customer: Optional[dict] = None) -> dict:
    """
    Compose a reply to a merchant/customer message within an existing conversation.

    Returns:
        dict with keys: action ("send"|"wait"|"end"), body?, cta?, rationale
    """
    identity = merchant.get("identity", {})
    voice = category.get("voice", {})
    hist_text = "\n".join(
        [f"  [{t.get('from', '?')}] {t.get('msg', t.get('body', ''))}" for t in conversation_history[-6:]]
    )

    prompt = f"""You are in a multi-turn WhatsApp conversation. Decide the BEST next action.

=== CONTEXT ===
CATEGORY: {category.get('slug', '?')} (voice: {voice.get('tone', '?')})
MERCHANT: {identity.get('name', '?')} ({identity.get('locality', '?')}, {identity.get('city', '?')})
LANGUAGES: {identity.get('languages', ['en'])}
TRIGGER that started this conversation: {trigger.get('kind', '?')}

=== CONVERSATION SO FAR ===
{hist_text}

=== MERCHANT'S LATEST MESSAGE ===
"{merchant_message}"

=== DETECTION RULES ===
1. AUTO-REPLY: If the message contains "thank you for contacting", "our team will respond", "automated assistant", or is the exact same text as a previous merchant message → treat as auto-reply
2. INTENT TRANSITION: If merchant says "yes", "let's do it", "ok", "go ahead", "sounds good", "proceed" → switch to ACTION mode immediately, provide concrete next steps
3. HOSTILE/OPT-OUT: If merchant says "stop", "not interested", "don't message", "unsubscribe", abuse → gracefully exit
4. OFF-TOPIC: If merchant asks about something outside Vera's scope (GST, legal, etc.) → politely decline, redirect to original thread
5. ENGAGED: If merchant asks a genuine question or shows interest → provide a helpful, specific answer

RESPOND WITH VALID JSON, one of three shapes:

For sending a reply:
{{"action": "send", "body": "your reply text", "cta": "binary_yes_no|open_ended|none", "rationale": "why"}}

For waiting (back off):
{{"action": "wait", "wait_seconds": 3600, "rationale": "why waiting"}}

For ending the conversation:
{{"action": "end", "rationale": "why ending"}}"""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"action": "send", "body": "Got it, let me look into that for you.",
                "cta": "open_ended", "rationale": f"LLM error fallback: {e}"}

    # Parse response
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        return {"action": "send", "body": raw[:300], "cta": "open_ended",
                "rationale": "Fallback — couldn't parse LLM JSON"}

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"action": "send", "body": raw[:300], "cta": "open_ended",
                "rationale": "Fallback — invalid JSON from LLM"}

    # Validate action
    if result.get("action") not in ("send", "wait", "end"):
        result["action"] = "send"

    if result["action"] == "send" and not result.get("body"):
        result["body"] = "Got it, let me check and get back to you."

    if result["action"] == "send" and not result.get("cta"):
        result["cta"] = "open_ended"

    if not result.get("rationale"):
        result["rationale"] = "Multi-turn reply"

    return result
