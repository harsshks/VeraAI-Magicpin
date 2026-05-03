"""
Vera AI Bot — FastAPI server for the magicpin AI Challenge.

Endpoints:
  GET  /v1/healthz   — liveness probe
  GET  /v1/metadata  — team identity
  POST /v1/context   — receive context pushes (category/merchant/customer/trigger)
  POST /v1/tick      — periodic wake-up; bot can initiate conversations
  POST /v1/reply     — receive merchant/customer replies; bot responds

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from composer import compose, compose_reply
from conversation_handlers import (
    ConversationState,
    is_auto_reply,
    is_hostile_or_optout,
    is_intent_commit,
    count_repeated_messages,
)

# ---------------------------------------------------------------------------
# App + State
# ---------------------------------------------------------------------------

app = FastAPI(title="Vera AI Bot", version="1.0.0")
START = time.time()

# In-memory stores
contexts: dict[tuple[str, str], dict] = {}          # (scope, context_id) → {version, payload}
conversations: dict[str, ConversationState] = {}     # conversation_id → ConversationState
suppressed_keys: set[str] = set()                    # suppression_keys already fired
ended_conversations: set[str] = set()                # conversation_ids that have ended
fired_triggers: set[str] = set()                     # trigger_ids already used to start convos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_context(scope: str, context_id: str) -> Optional[dict]:
    """Retrieve a stored context payload."""
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _count_contexts(scope: str) -> int:
    """Count how many contexts of a given scope are stored."""
    return sum(1 for (s, _) in contexts if s == scope)


def _get_category_for_merchant(merchant: dict) -> Optional[dict]:
    """Look up the CategoryContext for a merchant."""
    slug = merchant.get("category_slug", "")
    return _get_context("category", slug)


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": {
            "category": _count_contexts("category"),
            "merchant": _count_contexts("merchant"),
            "customer": _count_contexts("customer"),
            "trigger": _count_contexts("trigger"),
        }
    }


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Vera Rebuild",
        "team_members": ["Harsh Kumar"],
        "model": "llama-3.3-70b-versatile (via Groq)",
        "approach": "Trigger-kind routed prompt composer with multi-turn conversation handling, auto-reply detection, intent transition, and hostile exit. Category voice matching with post-LLM validation.",
        "contact_email": "harsh@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextBody):
    # Validate scope
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope",
                "details": f"scope must be one of {valid_scopes}"}

    key = (body.scope, body.context_id)
    existing = contexts.get(key)

    # Idempotency: reject stale or equal versions
    if existing and existing["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version",
                "current_version": existing["version"]}

    # Store atomically
    contexts[key] = {"version": body.version, "payload": body.payload}

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    for trg_id in body.available_triggers:
        # Skip already-fired triggers
        if trg_id in fired_triggers:
            continue

        # Get trigger context
        trigger = _get_context("trigger", trg_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id", "")
        merchant = _get_context("merchant", merchant_id)
        if not merchant:
            continue

        category = _get_category_for_merchant(merchant)
        if not category:
            continue

        # Check suppression
        supp_key = trigger.get("suppression_key", "")
        if supp_key and supp_key in suppressed_keys:
            continue

        # Get customer if customer-scoped
        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = _get_context("customer", customer_id)

        # Compose the message
        try:
            result = compose(category, merchant, trigger, customer)
        except Exception as e:
            continue

        if not result or not result.get("body"):
            continue

        # Generate conversation ID
        conv_id = f"conv_{merchant_id}_{trg_id}"

        # Skip if this conversation already exists
        if conv_id in ended_conversations:
            continue

        # Determine template name based on trigger kind
        kind = trigger.get("kind", "generic")
        send_as = result.get("send_as", "vera")
        template_name = f"vera_{kind}_v1" if send_as == "vera" else f"merchant_{kind}_v1"

        # Build template params
        identity = merchant.get("identity", {})
        merchant_name = identity.get("owner_first_name", identity.get("name", ""))
        template_params = [merchant_name, result.get("body", "")[:160], ""]

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": template_name,
            "template_params": template_params,
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", supp_key),
            "rationale": result.get("rationale", ""),
        }

        actions.append(action)

        # Track this trigger as fired
        fired_triggers.add(trg_id)
        if supp_key:
            suppressed_keys.add(supp_key)

        # Create conversation state
        state = ConversationState(conv_id, merchant_id, trg_id, customer_id)
        state.add_turn("vera", result["body"])
        conversations[conv_id] = state

        # Limit actions per tick to avoid timeout
        if len(actions) >= 5:
            break

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id

    # Get or create conversation state
    state = conversations.get(conv_id)
    if not state:
        state = ConversationState(conv_id, body.merchant_id or "", "", body.customer_id)
        conversations[conv_id] = state

    # Check if conversation already ended
    if state.is_ended or conv_id in ended_conversations:
        return {
            "action": "end",
            "rationale": "Conversation was previously ended."
        }

    # Add the incoming turn
    state.add_turn(body.from_role, body.message)

    # --- Rule-based detection (fast, no LLM needed) ---

    # 1. Auto-reply detection
    auto_reply_action = state.get_auto_reply_action(body.message)
    if auto_reply_action:
        if auto_reply_action["action"] == "end":
            ended_conversations.add(conv_id)
        return auto_reply_action

    # 2. Intent transition
    if is_intent_commit(body.message):
        result = {
            "action": "send",
            "body": "Great! Let's proceed. I will set this up for you. Reply CONFIRM to send the message.",
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant explicitly committed; switching to action mode."
        }
        state.add_turn("vera", result["body"])
        return result

    # 3. Hostile / opt-out
    if is_hostile_or_optout(body.message):
        result = state.handle_hostile()
        ended_conversations.add(conv_id)
        return result

    # --- LLM-powered reply (for genuine engagement) ---

    # Look up contexts
    merchant_id = body.merchant_id or state.merchant_id
    merchant = _get_context("merchant", merchant_id) or {}
    category = _get_category_for_merchant(merchant) or {}

    # Get trigger context
    trigger_id = state.trigger_id
    trigger = _get_context("trigger", trigger_id) or {}

    # Get customer if available
    customer = None
    customer_id = body.customer_id or state.customer_id
    if customer_id:
        customer = _get_context("customer", customer_id)

    # Compose reply via LLM
    try:
        result = compose_reply(
            category=category,
            merchant=merchant,
            trigger=trigger,
            conversation_history=state.turns,
            merchant_message=body.message,
            customer=customer,
        )
    except Exception as e:
        result = {
            "action": "send",
            "body": "Got it, let me look into that for you.",
            "cta": "open_ended",
            "rationale": f"Error in reply composition: {str(e)[:100]}",
        }

    # Track the bot's response in conversation state
    if result.get("action") == "send" and result.get("body"):
        state.add_turn("vera", result["body"])

    if result.get("action") == "end":
        ended_conversations.add(conv_id)

    return result


# ---------------------------------------------------------------------------
# POST /v1/teardown (optional — wipe state at end of test)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    """Wipe all state. Called by judge at end of test."""
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    ended_conversations.clear()
    fired_triggers.clear()
    return {"status": "wiped"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
