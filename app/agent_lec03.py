"""第3回の学習支援エージェント（用語集を使う版）。第5回で app/agent.py が書き換わった後も使えるように残す。

第3回の演習・宿題で作った形のまま：
- 指示文（INSTRUCTIONS）は、第3回の app/agent.py と一字一句同じ。
- ツールは用語集の search_glossary・get_lecture_summary と、第3回で足した make_quiz・承認付きの save_learning_log。登録の順も同じ。

第3回・第4回の比較用スクリプト（lectures/03/compare.py・lectures/04/compare.py）は、このエージェントで実験した。
モデルの作り方（build_model）・承認の流れ（run_with_approval）・trace の表示（print_trace）は、app/agent.py のものを使う。
"""

from pydantic_ai import Agent, DeferredToolRequests

from app import tools
from app.agent import ask_approval, build_model, print_trace, run_with_approval  # noqa: F401（比較用スクリプトがここから使えるように）

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""

# output_type に DeferredToolRequests を足すのは、承認が必要なツールが呼ばれたとき、実行を止めて承認待ちを返せるようにするため
agent = Agent(instructions=INSTRUCTIONS, output_type=[str, DeferredToolRequests])
agent.tool_plain(tools.search_glossary)
agent.tool_plain(tools.get_lecture_summary)
agent.tool_plain(tools.make_quiz)
agent.tool_plain(tools.save_learning_log, requires_approval=True)
