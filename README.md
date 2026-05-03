# Vera AI Bot — magicpin AI Challenge Submission

## Approach

Built a **trigger-kind-routed LLM composer** powered by **Groq (Llama-3.3-70b)** that takes magicpin's 4-context framework (Category, Merchant, Trigger, Customer) and produces personalized WhatsApp messages.

### Architecture

```
bot.py (FastAPI)
  ├── /v1/context    → version-based idempotent storage
  ├── /v1/tick       → trigger evaluation → composer.py
  └── /v1/reply      → conversation_handlers.py → composer.py

composer.py
  ├── dispatch_by_kind() → 10+ specialized prompt builders
  ├── build_context_block() → structured context injection
  └── validate_output() → CTA/language/taboo checks

conversation_handlers.py
  ├── Auto-reply detection (pattern + repetition)
  ├── Intent transition (commit → action mode)
  ├── Hostile/opt-out (graceful exit)
  └── Conversation state tracking
```

### Key Design Decisions

1. **Trigger-kind routing over single-prompt**: Different trigger types need fundamentally different framing (research digest ≠ performance dip ≠ recall reminder). Routing to specialized prompt builders produces category-appropriate output more reliably than a single universal prompt.

2. **Rule-based detection before LLM**: Auto-reply detection, hostile handling, and intent transition use regex patterns — faster, deterministic, and more reliable than asking the LLM every time. Only genuine engagement messages go to the LLM for reply composition.

3. **Escalating auto-reply backoff**: 1st auto-reply → one explicit prompt. 2nd → wait 24h. 3rd → end conversation. Avoids burning turns on canned replies.

4. **Hindi-English code-mix**: The system prompt encourages natural code-mixing when the merchant's language list includes "hi". The LLM handles this naturally given the instruction.

5. **Suppression key dedup**: Prevents the same message from being sent twice, using the trigger's suppression_key.

### Tradeoffs

- **Speed vs. quality**: Used Llama-3.3-70b via Groq for the best quality-to-latency ratio. A frontier model (Claude/GPT-4o) would score higher on nuance but risks the 30s timeout.
- **In-memory state**: Simple dict storage. Fine for test windows but wouldn't survive a restart. Production would use Redis.
- **Conservative sending**: The bot sends at most 5 actions per tick to stay within timeout. Could be tuned higher with faster LLM responses.

### What Additional Context Would Help Most

1. **Real merchant reply patterns** — anonymized transcripts of how merchants actually respond (beyond the 4 patterns in the brief)
2. **Suppression history** — knowing which messages the merchant has already received in the past 30 days would prevent topic fatigue
3. **Time-of-day preferences** — when merchants are most responsive (morning, afternoon, evening) for optimal send timing
4. **Offer performance data** — which offers actually drove conversions, not just which are active
