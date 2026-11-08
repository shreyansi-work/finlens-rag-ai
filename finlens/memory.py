"""Conversation memory.

Short-term: the last N turns. Long-term: a rolling summary plus user-level
facts (e.g. "tracks IT sector"). Memory only feeds query contextualization;
it is never dumped raw into the answer prompt.
"""
from collections import Counter
from dataclasses import dataclass, field

from finlens import llm
from finlens.text import company_name, find_tickers

IT_TICKERS = {"ARVL", "NILCS", "SAHSW", "INFY", "TCS", "WIPRO"}


@dataclass
class Memory:
    max_turns: int = 6
    turns: list = field(default_factory=list)          # [{"role", "content"}]
    summary: str = ""
    ticker_counts: Counter = field(default_factory=Counter)

    def add(self, question: str, answer: str):
        self.turns += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        self.ticker_counts.update(find_tickers(question))
        if len(self.turns) > self.max_turns * 2:
            old, self.turns = self.turns[:2], self.turns[2:]
            self.summary = self._summarise(old)

    def _summarise(self, old: list) -> str:
        if llm.enabled():
            return llm.complete(f"Update this running summary of a research session.\nSummary: {self.summary}\n"
                                f"New turns: {old}\nOutput 2-3 sentences.")
        return (self.summary + f" Asked: {old[0]['content']}").strip()[-600:]

    @property
    def facts(self) -> list[str]:
        out = []
        if self.ticker_counts and sum(self.ticker_counts[t] for t in IT_TICKERS) >= 2:
            out.append("tracks IT sector")
        if self.ticker_counts:
            out.append("most asked about " + company_name(self.ticker_counts.most_common(1)[0][0]))
        return out

    def history(self) -> list[dict]:
        return list(self.turns)


_sessions: dict[str, Memory] = {}


def get(session_id: str) -> Memory:
    return _sessions.setdefault(session_id, Memory())
