# Conversation storage for `chat_devstral.py`

Add persistent storage for chat conversations with full-text search across all
historical messages, plus auto-generated titles and rolling summaries
maintained by a background thread.

## Decisions (already settled)

- **Summarization cadence:** Background thread, fire-and-forget. Title after
  turn 1, summary refresh every 5 turns. Chat path never blocks on
  summarization.
- **DB location:** Repo-local at `./chat.db` (next to `chat_devstral.py`),
  gitignored.
- **Reasoning storage:** Separate `reasoning` column on `messages`. `/search`
  accepts `--in answer|reasoning|both` (default `both`).
- **Delivery:** Single PR.

## Files touched

| File | Change |
| --- | --- |
| `chat_storage.py` | **New.** ~250 lines. `ConversationStore` + `Summarizer`. |
| `chat_devstral.py` | Wire storage in; split reasoning from content in `_stream_response`; add slash commands; add `--resume`. |
| `.gitignore` | Add `chat.db*`. |

No new dependencies — `sqlite3` is stdlib and ships with FTS5 on macOS's
bundled SQLite.

## Schema

DB lives at `Path(__file__).parent / "chat.db"`. `CREATE TABLE IF NOT EXISTS`
runs on every connect; a `_schema_version` table records `version = 1` for
future migrations.

```sql
CREATE TABLE conversations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,                 -- ISO 8601 UTC
    updated_at            TEXT NOT NULL,
    model                 TEXT,
    host                  TEXT,                          -- "host:port"
    title                 TEXT,                          -- NULL until backfilled
    summary               TEXT,                          -- NULL until turn >= 5
    summary_through_turn  INTEGER NOT NULL DEFAULT 0,
    turn_count            INTEGER NOT NULL DEFAULT 0,
    prompt_tokens         INTEGER NOT NULL DEFAULT 0,
    completion_tokens     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE messages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id   INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn              INTEGER NOT NULL,
    role              TEXT NOT NULL,                     -- 'user' | 'assistant'
    content           TEXT NOT NULL DEFAULT '',          -- the visible answer
    reasoning         TEXT NOT NULL DEFAULT '',          -- chain-of-thought (assistant only)
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);

CREATE INDEX idx_messages_conv ON messages(conversation_id, turn);

CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    reasoning,
    content='messages',
    content_rowid='id'
);

-- FTS sync triggers (standard external-content pattern)
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, reasoning)
    VALUES (new.id, new.content, new.reasoning);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, reasoning)
    VALUES ('delete', old.id, old.content, old.reasoning);
END;
CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, reasoning)
    VALUES ('delete', old.id, old.content, old.reasoning);
    INSERT INTO messages_fts(rowid, content, reasoning)
    VALUES (new.id, new.content, new.reasoning);
END;

CREATE TABLE _schema_version (version INTEGER PRIMARY KEY);
INSERT INTO _schema_version VALUES (1);
```

PRAGMAs at connect time: `journal_mode=WAL`, `synchronous=NORMAL`,
`foreign_keys=ON`. WAL keeps the chat path snappy and lets the background
summarizer write without blocking turn appends.

## `chat_storage.py`

### `ConversationStore`

Owns the SQLite connection. Single connection, used from the chat thread and
the summarizer thread — guard with `threading.Lock` since `sqlite3.Connection`
is not safe for concurrent writes by default. (Alternative: open a second
connection in the summarizer; pick whichever is simpler — lock is fine here
because contention is near zero.)

```python
class ConversationStore:
    def __init__(self, db_path: Path): ...
    def start_conversation(self, model: str, host: str) -> int
    def append_turn(
        self,
        conversation_id: int,
        turn: int,
        user_message: str,
        assistant_content: str,
        assistant_reasoning: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None
    def update_title(self, conversation_id: int, title: str) -> None
    def update_summary(self, conversation_id: int, summary: str, through_turn: int) -> None
    def list_conversations(self, limit: int = 20) -> list[sqlite3.Row]
    def load_conversation(self, conversation_id: int) -> tuple[sqlite3.Row, list[sqlite3.Row]]
    def most_recent_id(self) -> Optional[int]
    def search(
        self,
        query: str,
        scope: Literal["answer", "reasoning", "both"] = "both",
        limit: int = 20,
    ) -> list[SearchHit]
```

`append_turn` runs as a single transaction: insert the user row, insert the
assistant row, update `conversations` rollups (`updated_at`, `turn_count`,
`prompt_tokens`, `completion_tokens`).

### Search

```python
@dataclass
class SearchHit:
    conversation_id: int
    title: Optional[str]
    updated_at: str
    message_id: int
    role: str
    turn: int
    snippet: str           # FTS5 snippet() with [bold]...[/bold] markers
    matched_in: str        # 'answer' | 'reasoning'
```

Implementation uses FTS5 column-filtered MATCH:
- `scope='answer'`   → `messages_fts MATCH 'content : ' || ?`
- `scope='reasoning'` → `messages_fts MATCH 'reasoning : ' || ?`
- `scope='both'`     → `messages_fts MATCH ?` (FTS5 searches all columns)

Use `snippet(messages_fts, -1, '[bold]', '[/bold]', '...', 12)` for the
preview. `matched_in` is derived in SQL by checking which column the highlight
landed in (CASE WHEN snippet on column 0 ≠ raw → 'answer' else 'reasoning'),
or simpler: run two queries when `scope='both'` and tag each result.

### `Summarizer`

Daemon thread + `queue.Queue`. Owns its own `requests.Session` and a reference
to the store. Never raises into the chat path.

```python
class Summarizer:
    def __init__(
        self,
        store: ConversationStore,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
    ): ...
    def enqueue_title(self, conversation_id: int, first_user: str, first_assistant: str) -> None
    def enqueue_summary(
        self,
        conversation_id: int,
        previous_summary: Optional[str],
        previous_through_turn: int,
        new_turns: list[tuple[str, str]],   # [(role, content), ...]
    ) -> None
    def shutdown(self, drain_timeout_s: float = 5.0) -> None
```

Job objects are simple dataclasses with a `kind` discriminator. Worker loop:

```python
while True:
    job = self.q.get()
    if job is SENTINEL: return
    try:
        if job.kind == 'title':
            title = self._call_vllm(TITLE_MESSAGES(job))
            title = _clean_title(title)
            self.store.update_title(job.conversation_id, title)
        elif job.kind == 'summary':
            summary = self._call_vllm(SUMMARY_MESSAGES(job))
            self.store.update_summary(job.conversation_id, summary, job.through_turn)
    except Exception as e:
        sys.stderr.write(f"[summarizer] {job.kind} failed: {e}\n")
```

`_call_vllm` issues a non-streaming `POST /v1/chat/completions`, reads
`choices[0].message.content`, and **discards any reasoning**. Title runs with
`max_tokens=40`, summary with `max_tokens=240`, both `temperature=0.3`.

`_clean_title` strips quotes, trailing punctuation, and any `<think>...</think>`
or `Title:` prefixes that smaller models occasionally emit. Truncate to 80 chars.

#### Prompts

**Title** (called once, after turn 1 lands):
```
system: You generate concise 4-6 word titles for chat conversations.
        Reply with ONLY the title text. No quotes. No trailing punctuation.
user:   USER: {first_user_message}

        ASSISTANT: {first_assistant_content}

        Title:
```

**Summary** (called after turns 5, 10, 15, ... — incremental):
```
system: You produce a 2-3 sentence summary of an ongoing chat conversation.
        Be concrete: name topics discussed and decisions reached. No preamble.
user:   Previous summary (covers turns 1-{previous_through_turn}):
        {previous_summary or "(none yet)"}

        New turns ({prev+1} through {current}):
        {formatted new turns}

        Updated summary covering all turns:
```

Incremental keeps prompt size bounded as conversations grow long.

### `shutdown` semantics

Push sentinel, `thread.join(timeout=drain_timeout_s)`. If still alive, log a
single warning and return — the daemon dies with the process. This means a
ctrl-C right after a turn may abandon an in-flight title; acceptable, the
conversation row still exists and will just show `(untitled)` in `/list`.

(Future enhancement, not in this PR: a lazy backfill in `/list` that enqueues
title generation for any visible row whose `title IS NULL`. Skip for v1.)

## Changes to `chat_devstral.py`

### Splitting reasoning from content in `_stream_response`

Currently both go into `response_content` (lines 285, 294). Split into two
buffers and change the return signature.

```python
# at top of _stream_response:
content_text = ""
reasoning_text = ""

# in the reasoning branch (currently line 285):
reasoning_text += reasoning

# in the content branch (currently line 294):
content_text += content

# return signature changes from:
#   return response_content, {...}
# to:
return content_text, reasoning_text, {
    "first_token_time": first_token_time,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
}
```

`chat()` updates accordingly:

```python
content_text, reasoning_text, metrics = self._stream_response(payload, start_time)
# ...
self.conversation_history.append({"role": "assistant", "content": content_text})
# (reasoning is NOT sent back to vLLM in subsequent turns — only persisted)
```

For the non-streaming branch (line 107+), assistant reasoning is `""` — vLLM's
non-stream response doesn't surface reasoning separately on the OpenAI route.
That's fine; reasoning persistence is a streaming-mode feature and the rest of
the system still works.

### Persistence in `chat()`

After a successful turn (current line ~163, where the metrics dict is built):

```python
if self.store is not None:
    if self.conversation_id is None:
        self.conversation_id = self.store.start_conversation(self.model, self.host_label)
    self.store.append_turn(
        conversation_id=self.conversation_id,
        turn=self.turn_count,
        user_message=user_message,
        assistant_content=content_text,
        assistant_reasoning=reasoning_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if self.summarizer is not None:
        if self.turn_count == 1:
            self.summarizer.enqueue_title(
                self.conversation_id, user_message, content_text,
            )
        if self.turn_count >= 5 and self.turn_count % 5 == 0:
            new_turns = self._collect_turns_since(self._summary_through_turn)
            self.summarizer.enqueue_summary(
                self.conversation_id,
                self._current_summary,
                self._summary_through_turn,
                new_turns,
            )
            self._summary_through_turn = self.turn_count
```

`self._summary_through_turn` and `self._current_summary` are tracked in the
chat object so we don't need a DB read per turn. They're refreshed when
`/load` runs.

DB write failures are caught here, logged, and swallowed — chat continues.

### `attach_storage(db_path)`

New method on `DevstralChat`. Called from `main()` after `check_server()` so
we have `self.model`. Constructs `ConversationStore` and `Summarizer` and
attaches. Skipping storage entirely is supported via `--no-store` flag (escape
hatch — not really needed but cheap to add and helps debugging).

### New slash commands (in `main()` dispatcher around line 610)

| Command | Behavior |
| --- | --- |
| `/list [N]` | Render Rich table: id, updated_at (relative: "3m ago"), title or `(untitled)`, turns, model. Default N=20. |
| `/load <id>` | Validate id is an int and exists. Replace `conversation_history` with `[{role, content} for m in messages]` (reasoning intentionally not replayed to vLLM). Reset `turn_count`, `summary_through_turn`, `current_summary` from the DB row. Print "Resumed conversation #N: <title>". Warn if conversation's stored model differs from current `self.model`. |
| `/search <query>` | Parse trailing `--in answer\|reasoning\|both`. Default `both`. Render hits grouped by conversation: title + date header, then snippet lines. |
| `/new` | Close current conversation (just null out `self.conversation_id`, `self.conversation_history`, `self.turn_count`). Old `/clear` becomes an alias. |
| `/title <text>` | Manually set title for active conversation. Error if no active conversation_id. |
| `/summary` | Print `self._current_summary` or `(none yet — generated every 5 turns)`. |

Update `/help` table to list these.

### `--resume` flag

```python
parser.add_argument(
    "--resume",
    nargs="?",
    const=-1,
    type=int,
    metavar="ID",
    help="Resume most recent conversation (no value), or specific conversation ID",
)
parser.add_argument(
    "--no-store",
    action="store_true",
    help="Disable conversation persistence (storage is on by default)",
)
```

After `attach_storage`:

```python
if args.resume is not None and chat.store is not None:
    target = args.resume if args.resume > 0 else chat.store.most_recent_id()
    if target:
        meta, msgs = chat.store.load_conversation(target)
        chat.conversation_history = [
            {"role": m["role"], "content": m["content"]} for m in msgs
        ]
        chat.conversation_id = target
        chat.turn_count = meta["turn_count"]
        chat._current_summary = meta["summary"]
        chat._summary_through_turn = meta["summary_through_turn"]
        # banner: "Resumed conversation #N: <title>  (turns: 7)"
```

### Shutdown wiring

In every exit path in `main()` (`/quit`, `KeyboardInterrupt`, `EOFError`):

```python
if chat.summarizer is not None:
    chat.summarizer.shutdown(drain_timeout_s=5.0)
```

Wrap the entire main loop in try/finally so shutdown runs even on uncaught
errors.

## `.gitignore`

Append:

```
# Local conversation history
chat.db
chat.db-journal
chat.db-wal
chat.db-shm
```

## Behavior with multiple models

`ConversationStore` records `model` and `host` per conversation. `/load` works
across models (the next turn just uses whatever model is currently connected),
but `/load` prints a warning if the stored model differs from the live one.
This matters because Devstral, GLM-4.7, and GLM-5.1 are all used through this
same client (see line 558 comment).

## Manual test plan

No test framework in this repo today, so verify by hand against a live vLLM
server (any of the presets):

1. Fresh DB created on first run; `chat.db` appears next to the script.
2. Send turn 1; ~1–2s later `sqlite3 chat.db "SELECT title FROM conversations"`
   shows a populated title.
3. Send 5 turns; verify `summary` is populated and `summary_through_turn = 5`.
   Send 5 more; verify summary refreshes and `summary_through_turn = 10`.
4. `/quit`, restart with `--resume`; verify history loads, turn counter
   continues, next turn appends to the same conversation_id.
5. `/list` shows recent conversations with titles and relative times.
6. `/search "<keyword>"` returns hits with bolded snippets across both
   columns.
7. Against GLM-5.1 (which emits reasoning):
   - Verify `reasoning` column populates for assistant rows.
   - `/search foo --in reasoning` finds matches only in chain-of-thought.
   - `/search foo --in answer` finds matches only in visible answers.
8. `/load <id>` mid-session swaps history; subsequent turns append to the
   loaded conversation.
9. `/load <id>` of a conversation from a different model prints a warning but
   still loads.
10. Ctrl-C immediately after sending a turn: process exits within ~5s, no
    hang.
11. With `--no-store`: no DB created, no slash commands for storage available
    (or they print "storage disabled").

## Out of scope (deferred)

- Lazy backfill of `(untitled)` rows when `/list` shows them.
- Tag/folder organization of conversations.
- Export to Markdown / JSON.
- Pruning or archival of old conversations.
- Sync across machines.
- Rich-rendered `--in` hint in `/help` per command.
