# 第4回 実験のメモ

自分で書くメモです。手順は [experiment.md](../experiment.md) にあります。書いたものは、Claude が answer.md に反映します。各見出しの下の説明は、書くことの目安です（書いたあとで消してもかまいません）。

## 1. 講師のラベルと違った項目と、そう判定した理由

`results/cli/compare_instructor.txt` を見て、講師と判定が違った項目（`#id 基準 自分=… 講師=…`）ごとに、自分はどう読んで、その判定にしたかを書く。違いが 0 件のときは、「迷った」を付けた項目と、迷った理由を書く。

## 2. 演習 STEP 3

STEP 2 の要確認から 1 つ選んで、方針を決めた記録を書く。

- 選んだ要確認の項目（件の id・基準・LLM が書いた理由）：
- 決めた方針（合格／不合格）：
- 理由：
- `data/eval/rubric.md` に書き足した文：

## 3. 実装

### 評価セットで直したこと

`data/eval/my_cases.jsonl` で、質問や期待するふるまいを直した件（id）と、その理由を書く。直さなかったときは、そう書く。

### ラベル付けで迷った件

迷った件の id・基準・迷った理由を書く。

## 4. 発展

読んだ問題ごとに、「自分なら解けるか」（解ける／解けない／たぶん）と、その理由（どこが難しいか、何を知っている必要があるか）を書く。SWE-bench Verified は、problem_statement だけを読んで書き、正解のパッチや PR は、そのあとで開く。

### SWE-bench Verified

- django__django-11099：
- astropy__astropy-12907：
- sympy__sympy-18199：
- sphinx-doc__sphinx-7590：

### τ²-bench

`user_scenario` とポリシーを読んで考えたあとで、`evaluation_criteria` を見る。

- airline 14：
- retail 24：
- telecom `[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]`（データ通信）：
- telecom `[service_issue]contract_end_suspension|unseat_sim_card[PERSONA:Hard]`（引き継ぎ）：

### OpenAI の記事で確かめたこと

手順書（段 6）の表の点（93 人・1,699 件・38.3%・61.1%・68.3%、2026-02-23 の記事の理由と 138 件の監査、2026-07-08 の撤回の文言）を、原文で確かめて書く。表の数字や日付と違っていたら、原文のとおりに書く。

- Introducing SWE-bench Verified（2024-08-13）：
- Why we no longer evaluate SWE-bench Verified（2026-02-23）：
- Separating signal from noise in coding evaluations（2026-07-08）：

## 5. 気づき・疑問

実験や、問題・記事を読んで気づいたこと、分からなかったことを書く。
