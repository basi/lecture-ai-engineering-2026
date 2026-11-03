"""第3回 発展（a）: 演習のエージェントを ADK（Google の Agent Development Kit）で組み立て直す。

使い方（リポジトリの直下で）:
    PYTHONPATH=. uv run --with google-adk==2.13.0 adk run adk_agent/glossary_agent

指示（INSTRUCTIONS）とツールは、Pydantic AI 版と同じものをそのまま使う。
変わるのは、エージェントを組み立てる部分だけ。承認付きのツール（save_learning_log）は入れない。
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL  # AIE_MODEL を切り替えたときも、Pydantic AI 版と同じモデルにそろえる

root_agent = Agent(
    model=MODEL,
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
