
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.assistant import Session, SupportAssistant

STATE_DIR = Path(__file__).resolve().parent / ".state"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="LearnForge support assistant (CLI)")
    parser.add_argument("message", help="The user's message")
    parser.add_argument("--session", default="default", help="Session id (persists last 12 local turns in .state/)")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved source IDs and scores")
    args = parser.parse_args()

    session = Session.load(args.session, STATE_DIR)
    assistant = SupportAssistant()
    result = assistant.handle(args.message, session=session)
    session.save(STATE_DIR)

    print(result.answer)
    print(f"\n[route: {result.route.value} | top_score: {result.gate.top_score:.3f}]")
    if args.show_sources:
        for s in result.sources:
            print(f"  - {s.record.id} ({s.record.source_type}) score={s.score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
