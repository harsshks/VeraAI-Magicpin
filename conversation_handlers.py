"""
Conversation Handlers — Multi-turn reply logic for Vera bot.

Handles:
  - Auto-reply detection (WhatsApp Business canned replies)
  - Intent transition detection (merchant commits to action)
  - Hostile/opt-out detection
  - Off-topic detection
  - Conversation state management
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Auto-reply detection
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    r"thank\s*you\s*for\s*contacting",
    r"our\s*team\s*will\s*respond",
    r"we\s*will\s*get\s*back",
    r"automated\s*(assistant|reply|message|response)",
    r"this\s*is\s*an?\s*auto(matic|mated)?\s*(reply|response|message)",
    r"out\s*of\s*(office|hours)",
    r"currently\s*(unavailable|busy)",
    r"aapki\s*jaankari.*shukriya",
    r"hamari\s*team\s*tak\s*pahuncha",
    r"we\s*are\s*currently\s*closed",
]

AUTO_REPLY_COMPILED = [re.compile(p, re.IGNORECASE) for p in AUTO_REPLY_PATTERNS]


def is_auto_reply(message: str) -> bool:
    """Detect if a message is a WhatsApp Business canned auto-reply."""
    msg_lower = message.lower().strip()

    # Pattern matching
    for pattern in AUTO_REPLY_COMPILED:
        if pattern.search(msg_lower):
            return True

    return False


def count_repeated_messages(conversation_history: list, message: str) -> int:
    """Count how many times this exact message appeared before in the conversation."""
    msg_stripped = message.strip().lower()
    count = 0
    for turn in conversation_history:
        prev_msg = turn.get("msg", turn.get("body", "")).strip().lower()
        if prev_msg == msg_stripped:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Intent transition detection
# ---------------------------------------------------------------------------

INTENT_COMMIT_PATTERNS = [
    r"\b(yes|yep|yea|yeah|haan|ha|ji)\b",
    r"\b(ok|okay|sure|alright|chalega|theek)\b",
    r"\b(let'?s\s*do\s*it|go\s*ahead|proceed|kar\s*do|karo)\b",
    r"\b(sounds?\s*good|perfect|great|done|agreed)\b",
    r"\b(start|begin|chalu|shuru)\b",
    r"\b(i\s*want|mujhe\s*chahiye|i'?d\s*like)\b",
    r"\b(confirm|approved?)\b",
]

INTENT_COMPILED = [re.compile(p, re.IGNORECASE) for p in INTENT_COMMIT_PATTERNS]

QUALIFYING_PATTERNS = [
    r"\bwhat\s*(if|about|would|do)\b",
    r"\bhow\s*(much|long|many|does)\b",
    r"\btell\s*me\s*more\b",
    r"\bcan\s*you\s*explain\b",
    r"\bwhat'?s\s*the\s*(cost|price|plan)\b",
]

QUALIFYING_COMPILED = [re.compile(p, re.IGNORECASE) for p in QUALIFYING_PATTERNS]


def is_intent_commit(message: str) -> bool:
    """Detect if the merchant is committing to action (saying yes/go ahead)."""
    msg_lower = message.lower().strip()

    # Check for qualifying questions — these override commit signals
    for pattern in QUALIFYING_COMPILED:
        if pattern.search(msg_lower):
            return False

    # Check for commit signals
    for pattern in INTENT_COMPILED:
        if pattern.search(msg_lower):
            return True

    return False


# ---------------------------------------------------------------------------
# Hostile / opt-out detection
# ---------------------------------------------------------------------------

HOSTILE_PATTERNS = [
    r"\b(stop|ruko|band\s*karo)\b.*\b(messag|send|bhejna)\b",
    r"\b(not\s*interested|nahi\s*chahiye)\b",
    r"\b(don'?t\s*(message|contact|disturb|bother)|mat\s*(bhejo|karo))\b",
    r"\b(unsubscribe|opt\s*out)\b",
    r"\b(spam|useless|bakwas|faltu|bekar)\b",
    r"\b(leave\s*me\s*alone|chhod\s*do)\b",
    r"\b(block|report)\b",
    r"\b(shut\s*up|chup)\b",
]

HOSTILE_COMPILED = [re.compile(p, re.IGNORECASE) for p in HOSTILE_PATTERNS]


def is_hostile_or_optout(message: str) -> bool:
    """Detect hostile messages or explicit opt-out requests."""
    msg_lower = message.lower().strip()

    for pattern in HOSTILE_COMPILED:
        if pattern.search(msg_lower):
            return True

    return False


# ---------------------------------------------------------------------------
# Off-topic detection
# ---------------------------------------------------------------------------

OFF_TOPIC_PATTERNS = [
    r"\b(gst|tax|income\s*tax|itr)\b",
    r"\b(legal|court|lawyer|advocate)\b",
    r"\b(loan|emi|credit|finance)\b",
    r"\b(personal|private)\b.*\b(question|matter)\b",
]

OFF_TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def is_off_topic(message: str) -> bool:
    """Detect if the merchant is asking about something outside Vera's scope."""
    msg_lower = message.lower().strip()

    for pattern in OFF_TOPIC_COMPILED:
        if pattern.search(msg_lower):
            return True

    return False


# ---------------------------------------------------------------------------
# Conversation state management
# ---------------------------------------------------------------------------

class ConversationState:
    """Tracks the state of a single conversation."""

    def __init__(self, conversation_id: str, merchant_id: str,
                 trigger_id: str = "", customer_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.merchant_id = merchant_id
        self.trigger_id = trigger_id
        self.customer_id = customer_id
        self.turns: list[dict] = []
        self.auto_reply_count = 0
        self.is_ended = False
        self.is_hostile = False

    def add_turn(self, from_role: str, message: str):
        """Add a turn to the conversation history."""
        self.turns.append({"from": from_role, "msg": message})

    def get_auto_reply_action(self, message: str) -> Optional[dict]:
        """Handle auto-reply escalation: detect → try once → wait → end."""
        if not is_auto_reply(message):
            # Also check for repeated messages
            repeats = count_repeated_messages(self.turns, message)
            if repeats < 2:
                return None
            # 2+ repeats of the same message = treat as auto-reply
            self.auto_reply_count = repeats

        self.auto_reply_count += 1

        if self.auto_reply_count == 1:
            # First auto-reply: try once more with a note
            return {
                "action": "send",
                "body": "Looks like an auto-reply. When the owner sees this -- just reply 'Yes' to continue from where we left off.",
                "cta": "binary_yes_no",
                "rationale": f"Detected auto-reply (count={self.auto_reply_count}); one explicit prompt to flag for the owner."
            }
        elif self.auto_reply_count == 2:
            # Second auto-reply: wait
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": f"Same auto-reply {self.auto_reply_count}x in a row → owner not at phone. Wait 24h before retry."
            }
        else:
            # 3+: end conversation
            self.is_ended = True
            return {
                "action": "end",
                "rationale": f"Auto-reply {self.auto_reply_count}x in a row, no real reply. Conversation has zero engagement signal; closing."
            }

    def handle_hostile(self) -> dict:
        """Handle hostile/opt-out message."""
        self.is_ended = True
        self.is_hostile = True
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out. Closing conversation; suppressing all triggers for this merchant."
        }
