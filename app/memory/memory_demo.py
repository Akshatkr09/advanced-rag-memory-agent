"""Run the assignment's cross-session semantic-memory demonstration.

Session 1 stores preferences for usr_99.
Session 2 searches those preferences using a different session_id.
"""
import json

from app.memory.mem0_store import Mem0Store


def run_demo():
    store = Mem0Store()
    user_id = "usr_99"

    first_session = [
        {
            "role": "user",
            "content": "Hello! I work in Python and prefer brief, code-only responses.",
        },
        {
            "role": "assistant",
            "content": "Got it. I'll remember your Python and response-style preferences.",
        },
    ]
    add_result = store.add_conversation(
        first_session, user_id=user_id, session_id="session_1"
    )
    print("Session 1 memory write:")
    print(json.dumps(add_result, indent=2, default=str))

    print("\nStored memories for usr_99:")
    print(json.dumps(store.get_user_memories(user_id), indent=2, default=str))

    context = store.build_user_context(
        "Write a quick merge sort routine.", user_id=user_id
    )
    print("\nSession 2 retrieved context:")
    print(context or "(No matching memories found)")


if __name__ == "__main__":
    run_demo()
