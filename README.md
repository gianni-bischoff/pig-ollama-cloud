# pig-ollama-cloud

A [PiG](https://github.com/MichaelKinsy/PiG) extension that adds **Ollama Cloud**
([ollama.com](https://ollama.com)) as an inference provider — with interactive
login and a live quota display in the footer.

## Features

- **Provider registration** — registers `ollama-cloud` speaking Ollama Cloud's
  OpenAI-compatible API (`openai-completions`). The model catalog is fetched
  live from the public `/v1/models` endpoint, so every model Ollama Cloud
  offers is selectable as `ollama-cloud/<model>`.
- **Interactive login** — `pig login ollama-cloud` prompts for your API key and
  stores it in PiG's credential store (`auth.json`). The key never appears in
  extension source. An environment variable is also supported for headless use.
- **Live quota display** — the footer shows how much of your session and weekly
  Ollama Cloud quota is left, refreshed every minute and after every model turn:

  ```
  oc session 38% · week 38% left
  ```

- **`ollama_cloud_status` tool** — check connectivity, the model catalog, and
  remaining quota on demand.

## Install

### Ambient (all sessions)

```bash
git clone https://github.com/gianni-bischoff/pig-ollama-cloud ~/.pig/agent/extensions/pig-ollama-cloud
pig install ~/.pig/agent/extensions/pig-ollama-cloud --validate-only
```

Restart PiG (or run `/reload`).

### One session at a time

```bash
pig -e /path/to/pig-ollama-cloud --model ollama-cloud/gpt-oss:120b
```

### Piglet

```yaml
name: my-app
extensions:
  - name: pig-ollama-cloud
    origins: [local:./extensions/pig-ollama-cloud]
model:
  provider: ollama-cloud
  name: gpt-oss:120b
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_CLOUD_API_KEY` | Cloud API key (https://ollama.com/cloud/settings) | — |
| `OLLAMA_API_KEY` | Fallback key name | — |
| `OLLAMA_CLOUD_BASE_URL` | Custom endpoint | `https://ollama.com` |

Precedence: key stored by `pig login ollama-cloud` → `OLLAMA_CLOUD_API_KEY` →
`OLLAMA_API_KEY`.

## Usage

```bash
# interactive login (recommended)
pig login ollama-cloud

# or headless
export OLLAMA_CLOUD_API_KEY=...

# pick a model
pig --model ollama-cloud/gpt-oss:120b
```

Notes:

- The stored login credential is resolved the same way a model request resolves
  it, so models appear in `/model` right after login, without any environment
  variable.
- `pig logout ollama-cloud` removes the stored key.
- The session-start registration upgrade fetches the live catalog with your
  resolved credential; the quota display needs a credential, the catalog does
  not.

## Development

PiG resolves the staged SDK requirement at build time, so `pig install
--validate-only` works out of the box. Running `go test` locally needs the
staged SDK; add a replace directive pointing at it:

```
replace github.com/MichaelKinsy/PiG/extensions/sdk => ~/.pig/state/pigsdk/sdk
```

## License

MIT
