"""
Persistent conversation storage for chat_devstral.py.

- ConversationStore: SQLite + FTS5. Persists conversations and messages,
  exposes full-text search across answer + reasoning columns.
- Summarizer: background thread that calls the connected vLLM server to
  generate titles (after turn 1) and rolling summaries (every 5 turns).
  Fire-and-forget — the chat path never blocks on a summarization request,
  and any error is logged to stderr instead of raised.

DB lives at ./chat.db next to this module and is gitignored.
"""

from __future__ import annotations

import queue
import sqlite3
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import requests


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    model                 TEXT,
    host                  TEXT,
    title                 TEXT,
    summary               TEXT,
    summary_through_turn  INTEGER NOT NULL DEFAULT 0,
    turn_count            INTEGER NOT NULL DEFAULT 0,
    prompt_tokens         INTEGER NOT NULL DEFAULT 0,
    completion_tokens     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id   INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn              INTEGER NOT NULL,
    role              TEXT NOT NULL,
    content           TEXT NOT NULL DEFAULT '',
    reasoning         TEXT NOT NULL DEFAULT '',
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, turn);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    reasoning,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, reasoning)
    VALUES (new.id, new.content, new.reasoning);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, reasoning)
    VALUES ('delete', old.id, old.content, old.reasoning);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, reasoning)
    VALUES ('delete', old.id, old.content, old.reasoning);
    INSERT INTO messages_fts(rowid, content, reasoning)
    VALUES (new.id, new.content, new.reasoning);
END;

CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO _schema_version VALUES (1);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / "chat.db"


# ----- Storage --------------------------------------------------------------


@dataclass
class SearchHit:
    conversation_id: int
    title: Optional[str]
    updated_at: str
    message_id: int
    role: str
    turn: int
    snippet: str
    matched_in: str  # 'answer' | 'reasoning'


class ConversationStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # The chat thread and the summarizer thread share one connection;
        # serialize with a lock. Contention here is essentially zero (a few
        # writes per turn vs. hours of idle time), so the simpler model wins.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we use explicit BEGIN/COMMIT
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def start_conversation(self, model: Optional[str], host: str) -> int:
        now = _now_iso()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO conversations (created_at, updated_at, model, host) "
                "VALUES (?, ?, ?, ?)",
                (now, now, model, host),
            )
            return cur.lastrowid

    def append_turn(
        self,
        conversation_id: int,
        turn: int,
        user_message: str,
        assistant_content: str,
        assistant_reasoning: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        now = _now_iso()
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    "INSERT INTO messages (conversation_id, turn, role, content, "
                    "reasoning, prompt_tokens, completion_tokens, created_at) "
                    "VALUES (?, ?, 'user', ?, '', 0, 0, ?)",
                    (conversation_id, turn, user_message, now),
                )
                self._conn.execute(
                    "INSERT INTO messages (conversation_id, turn, role, content, "
                    "reasoning, prompt_tokens, completion_tokens, created_at) "
                    "VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?)",
                    (
                        conversation_id, turn, assistant_content, assistant_reasoning,
                        prompt_tokens, completion_tokens, now,
                    ),
                )
                self._conn.execute(
                    "UPDATE conversations SET updated_at = ?, turn_count = ?, "
                    "prompt_tokens = prompt_tokens + ?, "
                    "completion_tokens = completion_tokens + ? WHERE id = ?",
                    (now, turn, prompt_tokens, completion_tokens, conversation_id),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def update_title(self, conversation_id: int, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )

    def update_summary(
        self, conversation_id: int, summary: str, through_turn: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET summary = ?, summary_through_turn = ? "
                "WHERE id = ?",
                (summary, through_turn, conversation_id),
            )

    def list_conversations(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, created_at, updated_at, model, title, turn_count "
                "FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    def load_conversation(
        self, conversation_id: int
    ) -> tuple[Optional[sqlite3.Row], list[sqlite3.Row]]:
        with self._lock:
            meta = self._conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if meta is None:
                return None, []
            msgs = self._conn.execute(
                "SELECT turn, role, content, reasoning FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
            return meta, msgs

    def most_recent_id(self) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            return row["id"] if row else None

    def get_summary(self, conversation_id: int) -> tuple[Optional[str], int]:
        """Return (summary_text, summary_through_turn) for a conversation."""
        with self._lock:
            row = self._conn.execute(
                "SELECT summary, summary_through_turn FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None, 0
        return row["summary"], row["summary_through_turn"]

    def get_turns_in_range(
        self, conversation_id: int, after_turn: int, through_turn: int
    ) -> list[tuple[str, str]]:
        """Return (role, content) pairs for turns in (after_turn, through_turn]."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = ? AND turn > ? AND turn <= ? "
                "ORDER BY id",
                (conversation_id, after_turn, through_turn),
            ).fetchall()
        return [(r["role"], r["content"]) for r in rows]

    def search(
        self,
        query: str,
        scope: Literal["answer", "reasoning", "both"] = "both",
        limit: int = 20,
    ) -> list[SearchHit]:
        # Quote the whole query as an FTS5 phrase so punctuation (hyphens,
        # parens, colons) in user input doesn't get parsed as FTS operators
        # or column refs. Trade-off: AND/OR/NEAR aren't usable from /search;
        # acceptable since this is a chat client, not a query language.
        phrase = '"' + query.replace('"', '""') + '"'
        if scope == "answer":
            fts_query = f"content : {phrase}"
        elif scope == "reasoning":
            fts_query = f"reasoning : {phrase}"
        else:
            fts_query = phrase
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.id AS message_id, m.conversation_id, m.role, m.turn,
                       c.title, c.updated_at,
                       snippet(messages_fts, 0, '[bold]', '[/bold]', '...', 12)
                           AS snip_content,
                       snippet(messages_fts, 1, '[bold]', '[/bold]', '...', 12)
                           AS snip_reasoning
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                JOIN conversations c ON c.id = m.conversation_id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        hits: list[SearchHit] = []
        for r in rows:
            snip_c = r["snip_content"] or ""
            snip_r = r["snip_reasoning"] or ""
            # Detect which column actually matched by looking for highlight
            # markers; snippet() on a non-matching column returns plain text.
            if "[bold]" in snip_c and scope != "reasoning":
                snippet, matched = snip_c, "answer"
            elif "[bold]" in snip_r and scope != "answer":
                snippet, matched = snip_r, "reasoning"
            elif scope == "reasoning":
                snippet, matched = snip_r, "reasoning"
            else:
                snippet, matched = snip_c, "answer"
            hits.append(SearchHit(
                conversation_id=r["conversation_id"],
                title=r["title"],
                updated_at=r["updated_at"],
                message_id=r["message_id"],
                role=r["role"],
                turn=r["turn"],
                snippet=snippet,
                matched_in=matched,
            ))
        return hits


# ----- Summarizer ----------------------------------------------------------


@dataclass
class _TitleJob:
    conversation_id: int
    first_user: str
    first_assistant: str


@dataclass
class _SummaryJob:
    conversation_id: int
    through_turn: int


_SHUTDOWN_SENTINEL = object()


class Summarizer:
    """Background thread for title + rolling-summary generation.

    Calls the connected vLLM server's `/v1/chat/completions` endpoint with
    small, bounded prompts. All errors are caught and logged to stderr so a
    failed summarization can never break the chat path.
    """

    def __init__(
        self,
        store: ConversationStore,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
    ):
        self.store = store
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s
        self._q: queue.Queue = queue.Queue()
        self._session = requests.Session()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue_title(
        self, conversation_id: int, first_user: str, first_assistant: str
    ) -> None:
        self._q.put(_TitleJob(conversation_id, first_user, first_assistant))

    def enqueue_summary(self, conversation_id: int, through_turn: int) -> None:
        """Request a summary that covers turns 1..through_turn.

        The worker reads the current `previous_summary` and the new turn range
        from the DB at job-execution time, so successive jobs (e.g. turn 5 and
        turn 10) chain correctly even if the turn-5 job hasn't completed by
        the time turn 10 lands.
        """
        self._q.put(_SummaryJob(conversation_id, through_turn))

    def shutdown(self, drain_timeout_s: float = 5.0) -> None:
        self._q.put(_SHUTDOWN_SENTINEL)
        self._thread.join(timeout=drain_timeout_s)
        if self._thread.is_alive():
            sys.stderr.write(
                f"[summarizer] drain timed out after {drain_timeout_s}s; "
                "abandoning in-flight job\n"
            )

    # ----- internals ------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is _SHUTDOWN_SENTINEL:
                return
            try:
                if isinstance(job, _TitleJob):
                    self._do_title(job)
                elif isinstance(job, _SummaryJob):
                    self._do_summary(job)
            except Exception as e:
                kind = type(job).__name__
                sys.stderr.write(f"[summarizer] {kind} failed: {e}\n")

    def _do_title(self, job: _TitleJob) -> None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate concise 4-6 word titles for chat "
                    "conversations. Reply with ONLY the title text. No "
                    "quotes. No trailing punctuation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"USER: {job.first_user}\n\n"
                    f"ASSISTANT: {job.first_assistant}\n\n"
                    "Title:"
                ),
            },
        ]
        text = self._call_vllm(messages, max_tokens=40)
        title = _clean_title(text)
        if title:
            self.store.update_title(job.conversation_id, title)

    def _do_summary(self, job: _SummaryJob) -> None:
        prev_summary, prev_through = self.store.get_summary(job.conversation_id)
        if prev_through >= job.through_turn:
            # An earlier-enqueued job (or a manual update) already covered
            # this range; nothing to do.
            return
        new_turns = self.store.get_turns_in_range(
            job.conversation_id, prev_through, job.through_turn
        )
        if not new_turns:
            return
        formatted = "\n".join(
            f"{role.upper()}: {content}" for role, content in new_turns
        )
        prev = prev_summary or "(none yet)"
        messages = [
            {
                "role": "system",
                "content": (
                    "You produce a 2-3 sentence summary of an ongoing chat "
                    "conversation. Be concrete: name topics discussed and "
                    "decisions reached. No preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous summary (covers turns 1-{prev_through}):\n"
                    f"{prev}\n\n"
                    f"New turns ({prev_through + 1} through {job.through_turn}):\n"
                    f"{formatted}\n\n"
                    "Updated summary covering all turns:"
                ),
            },
        ]
        text = self._call_vllm(messages, max_tokens=240)
        text = (text or "").strip()
        if text:
            self.store.update_summary(job.conversation_id, text, job.through_turn)

    def _call_vllm(self, messages: list[dict], max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        # Discard reasoning if any; only the visible content is the title/summary.
        return data["choices"][0]["message"].get("content") or ""


def _clean_title(text: str) -> str:
    """Strip the noise that small models occasionally emit around a title."""
    if not text:
        return ""
    # Some models leak chain-of-thought via <think>...</think> in content.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    for prefix in ("Title:", "title:", "TITLE:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) >= 2 and text[0] in ('"', "'", "`") and text[-1] == text[0]:
        text = text[1:-1].strip()
    text = text.rstrip(".!?;:,")
    text = text.splitlines()[0] if text else ""
    return text[:80]
