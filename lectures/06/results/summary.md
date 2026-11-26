# 第6回 実行結果のまとめ

`run.py --summarize` が、results/ の結果から作った（サーバーも API も使っていない）。

## 1. 段 B: [tool call] の行（cli/*.txt）

### step2_t1

- 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 1（glossary だけ）
- 開始: 2026-11-27T01:36:38+09:00／所要: 10.0 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   glossary_search_glossary({'query': '3レベル'})
```

### step2_t2

- 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 1（glossary だけ）
- 開始: 2026-11-27T01:36:53+09:00／所要: 9.8 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   glossary_search_glossary({'query': '3レベル'})
```

### step3_t1

- 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 3（glossary と fetch）
- 開始: 2026-11-27T01:37:08+09:00／所要: 9.6 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   fetch_fetch({'url': 'https://modelcontextprotocol.io/'})
```

### step3_t2

- 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 3（glossary と fetch）
- 開始: 2026-11-27T01:37:22+09:00／所要: 9.9 秒／終了コード: 1／やり直し: 0 回／結果: 停止（UnexpectedModelBehavior）

```text
なし（[tool call] の行がありません）
```

### step4_t1

- 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 4（glossary・fetch・aws）
- 開始: 2026-11-27T01:37:37+09:00／所要: 16.5 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   aws_aws___search_documentation({'search_phrase': 'Amazon S3 バージョニング 概要', 'topics': ['general']})
[tool call]   aws_aws___search_documentation({'topics': ['general'], 'search_phrase': 'Using versioning in S3 buckets'})
```

### step4_t2

- 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 4（glossary・fetch・aws）
- 開始: 2026-11-27T01:37:59+09:00／所要: 23.4 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   aws_aws___search_documentation({'search_phrase': 'Amazon S3 バージョニング 概要'})
[tool call]   aws_aws___search_documentation({'search_phrase': '"What is versioning" Amazon S3'})
```

### basic_t1

- 状態: S1（基礎の宿題を終えた app/mcp_server.py）／設定: STEP 4（glossary・fetch・aws）
- 開始: 2026-11-27T01:38:27+09:00／所要: 8.4 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   glossary_get_lecture_summary({'lecture': 2})
```

### basic_t2

- 状態: S1（基礎の宿題を終えた app/mcp_server.py）／設定: STEP 4（glossary・fetch・aws）
- 開始: 2026-11-27T01:38:40+09:00／所要: 8.1 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   glossary_get_lecture_summary({'lecture': 2})
```

### impl_t1

- 状態: S2（実装の宿題を終えた app/mcp_server.py（HEAD と同じ））／設定: STEP 4（glossary・fetch・aws）
- 開始: 2026-11-27T01:38:54+09:00／所要: 11.4 秒／終了コード: 0／やり直し: 0 回／結果: 成功

```text
[tool call]   glossary_make_quiz({'lecture': 99})
```

## 2. 段 A: 直接の呼び出し（条件 × lecture）

| 条件 | lecture | isError | 本文（先頭 80 字） |
|---|---|---|---|
| return | 99 | False | 第99回はありません。この講座は第1回〜第13回です。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。 |
| return | 5 | False | 第5回の用語は、まだ用語集にありません。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。 |
| return | 2 | False | {"question": "「学習率」の意味は？", "answer": "1回の更新でパラメータをどれだけ動かすかを決めるハイパーパラメータ。大きすぎると学習 |
| valueerror | 99 | True | Error executing tool make_quiz |
| valueerror | 5 | True | Error executing tool make_quiz |
| valueerror | 2 | False | {"question": "「Hugging Face Hub」の意味は？", "answer": "公開されたモデルやデータセットを共有・ダウンロードできるプ |
| toolerror | 99 | True | Error executing tool make_quiz: 第99回はありません。この講座は第1回〜第13回です。確認問題を出せる回: 1, 2, 3, 4 |
| toolerror | 5 | True | Error executing tool make_quiz: 第5回の用語は、まだ用語集にありません。確認問題を出せる回: 1, 2, 3, 4, 6。利用者 |
| toolerror | 2 | False | {"question": "「ファインチューニング」の意味は？", "answer": "事前学習済みモデルを、特定のタスクや形式に合ったデータで追加学習させる |
| schema | 99 | True | Error executing tool make_quiz: 1 validation error for make_quizArguments lectur |
| schema | 5 | False | 第5回の用語は、まだ用語集にありません。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。 |
| schema | 2 | False | {"question": "「ファインチューニング」の意味は？", "answer": "事前学習済みモデルを、特定のタスクや形式に合ったデータで追加学習させる |

### lecture=99 のとき、モデルに渡る文（FunctionModel の実行で残ったメッセージ）

- return（ToolReturnPart の content）:

```text
第99回はありません。この講座は第1回〜第13回です。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。
```

- valueerror（RetryPromptPart の model_response()）:

```text
Error executing tool make_quiz

Fix the errors and try again.
```

- toolerror（RetryPromptPart の model_response()）:

```text
Error executing tool make_quiz: 第99回はありません。この講座は第1回〜第13回です。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。

Fix the errors and try again.
```

- schema（RetryPromptPart の model_response()）:

```text
Error executing tool make_quiz: 1 validation error for make_quizArguments
lecture
  Input should be less than or equal to 13 [type=less_than_equal, input_value=99, input_type=int]
    For further information visit https://errors.pydantic.dev/2.14/v/less_than_equal

Fix the errors and try again.
```

## 3. 段 C: 条件 × 質問 × 試行

| 条件 | 質問 | 試行 | make_quiz の引数 | エラー数 | status | 回答（先頭 60 字） | 入力 tokens | 出力 tokens | リクエスト |
|---|---|---|---|---|---|---|---|---|---|
| return | q99 | 1 | [] | 0 | ok | 指定された「第99回」の講義は存在しません。本講座（AIE2026）は全何回構成か、あるいは何回目までの内容を確認したい | 456 | 45 | 1 |
| return | q99 | 2 | [] | 0 | ok | 申し訳ありません。第99回という回は設定されておらず、講義データが存在しません。 対応している講義の範囲内でクイズを作成 | 456 | 72 | 1 |
| return | q99 | 3 | [] | 0 | ok | 指定された第99回は、この講座の範囲外（または存在しない回）となります。 講座は第何回まででしょうか？正しい回の番号を指 | 456 | 45 | 1 |
| return | q5 | 1 | [5] | 0 | ok | 第5回の用語はまだ用語集に登録されていないため、第5回の確認問題を作ることができません。 もしよろしければ、現在確認問題 | 990 | 97 | 2 |
| return | q5 | 2 | [5] | 0 | ok | 第5回の用語はまだ用語集に登録されていないため、確認問題を作成することができません。 現在、確認問題を出せる回は以下の通 | 990 | 101 | 2 |
| return | q5 | 3 | [5] | 0 | ok | 第5回は、現在のところ用語集に用語が登録されていないため、確認問題を作成することができません。 よろしければ、現在確認問 | 990 | 84 | 2 |
| valueerror | q99 | 1 | [] | 0 | ok | 申し訳ありませんが、第99回の講義は設定されていません。（本講座は全何回か、または該当する回の内容をご確認ください） も | 456 | 58 | 1 |
| valueerror | q99 | 2 | [] | 0 | ok | 第99回の講義は設定されていないため、用語集や確認問題を作成することができません。 もし他の回（例えば第1回〜最終回など | 456 | 50 | 1 |
| valueerror | q99 | 3 | [] | 0 | ok | 第99回という回は講座（AIE2026）の講義リストに存在しないため、用語集や講義データが見つかりませんでした。 AIE | 456 | 84 | 1 |
| toolerror | q99 | 1 | [] | 0 | ok | 第99回という回は講座（AIE2026）の範囲に含まれていないため、用語集にデータがありません。そのため、第99回のクイ | 456 | 74 | 1 |
| toolerror | q99 | 2 | [] | 0 | ok | 第99回という回は、この講座（AIE2026）の範囲には含まれていないようです。 当講座のカリキュラムは第1回から第6回 | 456 | 83 | 1 |
| toolerror | q99 | 3 | [] | 0 | ok | 申し訳ありませんが、第99回の講義は存在しません。本講座（AIE2026）は全何回構成か、またはどの回のクイズをご希望か | 456 | 44 | 1 |
| schema | q99 | 1 | [] | 0 | ok | 申し訳ありません。この講座（AIE2026）は全13回のため、第99回は存在しません。 第1回から第13回までの間で、ク | 465 | 49 | 1 |
| schema | q99 | 2 | [] | 0 | ok | 申し訳ありませんが、この講座（AIE2026）は全回数が設定されており、「第99回」という講義は存在しません。（通常の講 | 465 | 106 | 1 |
| schema | q99 | 3 | [] | 0 | ok | 申し訳ありませんが、この講座（AIE2026）は第13回までの構成となっており、第99回は存在しません。第1回から第13 | 465 | 62 | 1 |

## 4. aws_ ツールの定義（step4_tools・basic_tools・impl_tools の一致）

- step4_tools: 5 個（aws_aws___read_documentation, aws_aws___search_documentation, aws_aws___list_regions, aws_aws___get_regional_availability, aws_aws___retrieve_skill）
- basic_tools: 5 個（aws_aws___read_documentation, aws_aws___search_documentation, aws_aws___list_regions, aws_aws___get_regional_availability, aws_aws___retrieve_skill）
- impl_tools: 5 個（aws_aws___read_documentation, aws_aws___search_documentation, aws_aws___list_regions, aws_aws___get_regional_availability, aws_aws___retrieve_skill）

一致していました（step4_tools, basic_tools, impl_tools）。

## 5. env_probe（子プロセスが見られる環境変数）

- クライアントに渡した環境変数の名前: AIE_LEC06_CANARY, HOME, LANG, PATH, PYTHONUNBUFFERED
- 目印の名前: AIE_LEC06_CANARY
- 子（サーバー）の環境変数の名前: HOME, LC_CTYPE, PATH
- 子の環境に目印があったか: False
- /proc があるか: True（linux）
- 親の /proc/PID/environ: environ を読めた: True／目印があった: True
- 先祖のプロセス: 4 段（environ を開けた: 4 段）
