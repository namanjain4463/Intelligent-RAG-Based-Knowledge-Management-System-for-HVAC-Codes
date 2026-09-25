"""Application compatibility entry point for the validated v2 ReAct runtime.

The legacy LangChain/legacy-schema agent is intentionally no longer imported.
The public query_agent/get_statistics names remain available to bot.py while
the implementation lives in v2_ingestion.react_runtime.
"""

from v2_ingestion.react_runtime import get_statistics, query_agent, query_agent_result

__all__ = ["query_agent", "query_agent_result", "get_statistics"]


if __name__ == "__main__":
    question = input("Question: ").strip()
    if question:
        print(query_agent(question))
