# finance-context-agent

[Русский](README.ru.md) · **English**

Answer financial questions over finance-context-builder slices.

A cash-flow workbook does not fit in a model context. This service answers one financial question by reading a book that [finance-context-builder](https://github.com/x0r1x/finance-context-builder) has already parsed. It loads the job passport and the period axes, searches rows by label, and sends the model only the observations those rows need. A number in the answer is a string from the Excel cache. Every such number matches a citation value or that citation's `period_id`. An empty cell and a `not_applicable` cell stay empty.

The questions the agent asks the user are in Russian, because the prompts are Russian.

## Contents

- [Where the workbook comes from](#where-the-workbook-comes-from)
- [Requirements](#requirements)
- [Environment](#environment)
- [Local run](#local-run)
- [Docker](#docker)
- [Ask a question](#ask-a-question)
- [HTTP API](#http-api)
- [When an answer is accepted](#when-an-answer-is-accepted)
- [Sessions](#sessions)
- [Checks](#checks)
- [Live check](#live-check)

## Where the workbook comes from

Excel is parsed by finance-context-builder. Upload stays on the parser, `POST /v1/context-jobs`, and the agent starts from a finished job. A question about a row calls the parser over HTTP and reads three slices: the summary passport, the catalog (labels and period axes, no cell cache), and observations (the cache for the chosen rows). A book overview also reads `GET /v1/context-jobs/{id}/context.json` and answers with the book's own sections and the stored figures of the live rows. Rows the parser marked excluded, and the workbook audit (formula counts, cache warnings, mapping totals), stay out of that text. The file is not sent to the model. `graph.json` stays on the parser. Formulas stay as the cache stored them. This repository does not import the `finance_context` package.

A row the parser left without a concept is still found by its label. Its `concept_id` stays empty.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/), or Docker.
- finance-context-builder listening on `http://127.0.0.1:8080`, with the job in `succeeded`.
- An OpenAI-compatible model on `http://127.0.0.1:1234/v1`. LM Studio on the host is the expected server.
- Redis 8 for dialog checkpoints. `docker compose` starts it. A host process needs a Redis you can reach at `REDIS_URL`.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARSER_BASE_URL` | `http://127.0.0.1:8080` | Parser. Inside the container a loopback host is rewritten to `host.docker.internal`. The base does not gain `/v1` |
| `PARSER_TIMEOUT_SEC` | `15` | Parser HTTP timeout, seconds. Greater than 0 |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | Model. Inside the container a loopback host is rewritten to `host.docker.internal` |
| `LLM_API_KEY` | empty | Sent as `Authorization: Bearer …` when set |
| `LLM_MODEL` | `local` | Model name sent to the LLM server |
| `LLM_TIMEOUT_SEC` | `60` | Model HTTP timeout, seconds. Greater than 0 |
| `LLM_TEMPERATURE` | `0` | Sampling temperature, from 0 to 2 |
| `REDIS_URL` | `redis://localhost:6379/0` | Dialog checkpoints. An unset Compose value is `redis://:devpassword@redis:6379/0`. `REDIS_PASSWORD` changes that password |
| `HOST` | `0.0.0.0` | Bind address of the process. Compose always sets `0.0.0.0` |
| `PORT` | `8090` | Bind port of the process |
| `CHECKPOINT_TTL_MINUTES` | `1440` | Idle lifetime of a checkpoint, at least 1 minute |
| `CHECKPOINT_REFRESH_ON_READ` | `true` | Reading a thread resets that lifetime |
| `LOCK_TTL_SEC` | `900` | How long a dead run holds a thread, at least 1 second |
| `RECURSION_LIMIT` | `40` | LangGraph step cap for one invoke, at least 1 |
| `CONTENT_BUDGET` | `4` | Content steps before a new gap closes the turn, at least 1 |
| `CLARIFY_BUDGET` | `2` | Clarification replies before the turn closes, at least 1 |
| `OBSERVATION_CAP` | `48` | Observations kept for one answer, from 1 to 48 |
| `CATALOG_SEARCH_LIMIT` | `32` | Catalog hits requested per needle, from 1 to 100 |
| `PRECEDENT_DEPTH` | `2` | Precedent and dependent graph depth, from 0 to 3 |
| `MAX_DEPENDENT_ROWS` | `4` | Dependent rows for an influence question, from 1 to 8 |
| `DEPENDENT_OBSERVATION_LIMIT` | `8` | Observations requested per dependent row, from 1 to 48 |
| `LANGGRAPH_STRICT_MSGPACK` | `true` | Strict checkpoint serialization |

Copy `.env.example` to `.env` to override a default. `uv run` and Compose both read that file. The file is optional for Compose. A blank value keeps the default. A value in the file replaces the default. A number outside the range in the table stops the process at startup. The process environment wins over `.env`. An unset `REDIS_URL` is `redis://:devpassword@redis:6379/0` inside Compose. `REDIS_PASSWORD` is the password of that Redis. Inside the container, loopback hosts of the parser and the model become `host.docker.internal`.

The public `model` field of a completion is always `finance-context-agent`. `LLM_MODEL` is only the name sent upstream.

## Local run

```bash
uv sync
uv run finance-context-agent
```

`uv sync` creates `.venv` from `uv.lock`. `uv run` uses that environment.

```bash
curl -s http://127.0.0.1:8090/healthz
curl -s http://127.0.0.1:8090/readyz
```

`GET /healthz` is process liveness: `{"status":"ok"}`. `GET /readyz` is 200 when Redis answers `PING` and the parser's own `/readyz` answers. Otherwise it is 503:

```json
{"status": "unavailable", "redis": false, "parser": true}
```

The route schema is also on [Swagger UI](http://127.0.0.1:8090/docs).

## Docker

Parser and LM Studio stay on the host. Compose runs the agent and Redis.

```bash
docker compose up --build
```

The API is published on `127.0.0.1:8090`. The process binds `PORT` (default 8090); the compose port mapping uses the same `PORT`. Redis is on the compose network only; port 6379 is not published. The compose password `devpassword` is the local default; `REDIS_PASSWORD` changes it. The parser and the model must listen on the host gateway, because the container calls `host.docker.internal`.

## Ask a question

`POST /v1/chat/completions`. The body is a chat completion plus `job_id` and `thread_id`. The same two values may be sent as `X-Job-Id` and `X-Thread-Id`. A non-empty body field wins over the header. An empty body field falls through to the header.

```bash
curl -sS http://127.0.0.1:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "job_id": "<job_id from the parser>",
    "thread_id": "dialog-1",
    "messages": [{"role": "user", "content": "Какой DSCR в 2030?"}]
  }'
```

`job_id` matches `^[A-Za-z0-9._-]{1,128}$`. `thread_id` matches `^[A-Za-z0-9_-]{1,255}$`. Omit `thread_id` and the agent mints a UUID and returns it. Omit `job_id` on a new dialog and the agent lists succeeded books by file name and waits:

```text
Какую книгу открыть?
- model.xlsx
```

Send the next user message on the same `thread_id` with the file name. A name it cannot match is asked once more: `Не нашёл такую книгу. Назовите файл ещё раз.`

A finished completion looks like this. `id` and `created` vary.

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "created": 0,
  "model": "finance-context-agent",
  "thread_id": "dialog-1",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "DSCR в 2030 — 1.25."},
      "finish_reason": "stop"
    }
  ],
  "awaiting_user": false,
  "satisfactory": true,
  "gaps": [],
  "citations": [
    {
      "row_key": "row-dscr",
      "period_id": "2030",
      "cell": "C10",
      "value": "1.25",
      "value_status": "cached"
    }
  ],
  "steps": []
}
```

`finish_reason` is always `stop`. A pause is `awaiting_user: true`, and `content` is the question. While that flag is true, the next user message on the same `thread_id` resumes the pause. After the graph has finished, the next message is a new question on the same dialog: the chosen book stays, and a follow-up such as `А почему?` uses the citations already stored. A different `job_id` on a thread that already has a book is 409 `job_mismatch`.

Several catalog rows, or a period that maps to more than one axis key, produce a question and no cell values. The row question lists labels and `total`:

```text
Какую строку взять?
«DSCR»: 2. DSCR observed [Debt, dscr]; DSCR covenant [Debt, dscr_covenant]
```

Reply with the label you want, on the same `thread_id`. Two replies is the cap (`CLARIFY_BUDGET`). The same question asked again closes the turn with `satisfactory: false`.

Other pauses, in the words the agent sends:

| Situation | Question |
| --- | --- |
| No catalog hit after a second search | `Такой строки нет. Назовите подпись иначе или выберите другую метрику.` |
| The period matches several axis keys | `Период подходит нескольким ключам оси. Назовите один.` |
| The period is not on the row axis | `Период не находится на оси строки. Назовите ключ или год.` |
| Chosen rows do not share one period | `Период не один на всех выбранных строках. Назовите ключ.` |
| Observations were truncated and no period was named | `Ряд обрезан лимитом 48. Назовите период, среднее по обрезанному ряду не считается.` |

`stream: true` is 400 `stream_unsupported`. The `model` field of the request is ignored.

## HTTP API

- `GET /healthz` — process liveness.
- `GET /readyz` — Redis and the parser. 503 when either is down.
- `POST /v1/chat/completions` — one turn, or a resume of a paused turn.

An error body is always `{"error": "<code>"}`.

| code | HTTP |
| --- | --- |
| `stream_unsupported` | 400 |
| `messages_required` | 400 |
| `last_message_not_user` | 400 |
| `user_message_required` | 400 |
| `bad_thread_id` | 400 |
| `bad_job_id` | 400 |
| `thread_busy` | 409 |
| `job_mismatch` | 409 |
| `upstream_unavailable` | 503 |

The last message must be `user`. `content` is a string, or a list of parts whose `text` fields are joined.

A book that is still building is a normal 200. The assistant text is `Книга ещё собирается.` and `satisfactory` is false. An unknown job id is `Книга не найдена.`, also 200. There is no succeeded book to choose from: `Готовых книг нет.` A network error, a timeout, or an HTTP 5xx from the parser or the model is 503 `upstream_unavailable`. The parser client waits 15 seconds (`PARSER_TIMEOUT_SEC`). The model client waits 60 seconds (`LLM_TIMEOUT_SEC`).

`gaps` names what failed. Examples: `report_not_ready`, `not_found`, `number:…`, `cell:…`, `scale:…`, `precedent`, `compare_sides`, `schema`, `clarify`. A citation carries `row_key`, `period_id`, `cell`, and when present `value` and `value_status` (`cached`, `empty`, `zero_explicit`, `not_applicable`). `steps` records the search and the observation calls of the turn.

## When an answer is accepted

`satisfactory` is true when the citation check passes. A gap that is only a missing scale word still publishes the answer.

The citation check requires every number in the assistant text to equal a citation `value` or `normalized_value`, or to match that citation's `period_id`. The same number may also appear in that observation's formula text or in a direct precedent's `value`, `normalized_value`, or `period_id`. The cell is one of the observations just retrieved. A scale other than 1 has to be named in the answer. A comparison names both sides. When the question asks how a row is calculated (`почему`, `из чего`, `как считается`, `какая формула`, `why`) and the observation has a formula text or a direct precedent, the answer quotes that formula text or a precedent cell. An empty or `not_applicable` citation passes with `value_status` and no number. A number the slice does not confirm is replaced with `Подтверждённого числа в срезе нет.` and the gap ids are appended.

A new code gap sends the agent back for another slice, up to four content steps (`CONTENT_BUDGET`). The same gap, the same clarification question, a plan the model cannot express as JSON, or a spent budget closes the turn with `satisfactory: false`. The agent does not ask whether the user is satisfied.

A series cut at 48 observations (`OBSERVATION_CAP`) is not averaged. The agent asks for a period. A question about what a row affects also retrieves dependent rows, inside the same cap of 48 observations.

## Sessions

One `thread_id` is one dialog. Threads do not share citations or the chosen book. Every agent replica reads the same Redis, so another process can continue a dialog. A checkpoint lives for 24 hours of idle time (`CHECKPOINT_TTL_MINUTES`) and the timer resets when the thread is read (`CHECKPOINT_REFRESH_ON_READ`). Memory does not cross threads.

One run at a time holds a thread. A second request receives 409 `thread_busy` and does not cancel the first. A run that dies releases the thread within 15 minutes (`LOCK_TTL_SEC`).

## Checks

```bash
uv run ruff check src tests
uv run pytest
```

The suite fakes the parser and the model. `tests/test_redis_checkpoint.py` starts Redis 8 in Docker, pauses a dialog, and resumes it on a new graph pointed at the same Redis. It is skipped when Docker is not installed.

### Live check

Redis, the parser, and LM Studio are already running, and the agent is already listening. `scripts/run.sh` sends HTTP only. The workbook stays on the parser. A question goes out only when `JOB_ID` is set.

```bash
bash scripts/run.sh
JOB_ID=<parser job_id> bash scripts/run.sh 'Какой DSCR в 2030?'
THREAD_ID=<thread_id from summary.txt> JOB_ID=<parser job_id> bash scripts/ask.sh 'Наблюдённый'
```

`bash scripts/run.sh` checks `GET /healthz`, `GET /readyz` (Redis and the parser both answer), and the 400 responses that return before the model: `stream_unsupported`, `messages_required`, `last_message_not_user`, `user_message_required`, `bad_thread_id`, and `bad_job_id` from the body and from `X-Job-Id`.

A question writes `completion.json`. The check requires HTTP 200, `object` `chat.completion`, `model` `finance-context-agent`, and `finish_reason` `stop`. `awaiting_user: true` is a successful check. The assistant text is the pause, and `satisfactory` stays with the book and the model. The third command sends the next user message on that `thread_id`. The server decides whether that resumes the pause or starts a new turn.

The run directory is `out/<timestamp>/`:

```text
healthz.json  readyz.json  summary.txt
error-stream.json  error-messages.json  error-role.json  error-empty.json
error-thread.json  error-job.json  error-header-job.json
ask.json  completion.json
```

Each error response sits next to the request that produced it (`*.request.json`). `ask.json` and `completion.json` are written only when `JOB_ID` is set. Override `BASE_URL` (default `http://127.0.0.1:8090`), `OUT_DIR` (default `./out`), `RUN_DIR`, `THREAD_ID`, `QUESTION`, and `CHAT_TIMEOUT_SEC` (default 180, one POST). A client timeout leaves the turn running. The thread stays locked until `LOCK_TTL_SEC` (default 900).
