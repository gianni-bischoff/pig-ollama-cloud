// Package pig_ollama_cloud registers Ollama Cloud (https://ollama.com) as a
// PiG inference provider and pins live quota usage to the footer.
//
// The provider speaks Ollama Cloud's OpenAI-compatible API. Authentication:
//
//	OLLAMA_CLOUD_API_KEY   Ollama Cloud API key; create one at
//	                       https://ollama.com/cloud/settings. OLLAMA_API_KEY
//	                       is accepted as a fallback. Interactive login through
//	                       `pig login ollama-cloud` stores the key in PiG's
//	                       credential store and takes precedence.
//	OLLAMA_CLOUD_BASE_URL  Optional custom endpoint (default https://ollama.com).
//
// The /v1/models catalog endpoint is public, so the model list is always
// fetched live; the API key is only required for completion requests and for
// the usage endpoint.
//
// The footer shows the remaining session and weekly quota of the configured
// account, next to PiG's own usage info.
package pig_ollama_cloud

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

// Provider identity and defaults.
const (
	providerID   = "ollama-cloud"
	providerName = "Ollama Cloud"
	defaultBase  = "https://ollama.com"

	// settingsURL is shown in the login prompt.
	settingsURL = "https://ollama.com/cloud/settings"
)

// Environment variables, in precedence order.
const (
	envAPIKey  = "OLLAMA_CLOUD_API_KEY"
	envAPIAlt  = "OLLAMA_API_KEY"
	envBaseURL = "OLLAMA_CLOUD_BASE_URL"
)

// Model catalog defaults. Ollama Cloud does not report per-model limits in
// /v1/models, so every entry uses conservative values.
const (
	contextWindow = 131072
	maxTokens     = 32768
	httpTimeout   = 5 * time.Second

	// usageInterval is how often the footer quota is refreshed.
	usageInterval = time.Minute
)

// errNoAPIKey reports that no credential is configured.
var errNoAPIKey = errors.New("no Ollama Cloud API key configured (run 'pig login ollama-cloud' or set OLLAMA_CLOUD_API_KEY)")

// client is shared by every call to Ollama Cloud.
var client = &http.Client{Timeout: httpTimeout}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// apiKey returns the configured key from the environment, or "" when absent.
func apiKey() string {
	for _, name := range []string{envAPIKey, envAPIAlt} {
		if key := strings.TrimSpace(os.Getenv(name)); key != "" {
			return key
		}
	}
	return ""
}

// baseURL returns the configured endpoint with a normalized scheme and no
// trailing slash.
func baseURL() string {
	base := strings.TrimRight(strings.TrimSpace(os.Getenv(envBaseURL)), "/")
	if base == "" {
		return defaultBase
	}
	if !strings.HasPrefix(base, "http://") && !strings.HasPrefix(base, "https://") {
		base = "https://" + base
	}
	return base
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

// getJSON performs a GET request and decodes the JSON body into out.
func getJSON(url, key string, out any) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Model catalog
// ---------------------------------------------------------------------------

type openAIModelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

// fetchModels queries Ollama Cloud's public /v1/models catalog. key is sent
// as the bearer when non-empty, but the catalog itself does not require one.
func fetchModels(base, key string) ([]map[string]any, error) {
	var list openAIModelsResponse
	if err := getJSON(base+"/v1/models", key, &list); err != nil {
		return nil, err
	}

	models := make([]map[string]any, 0, len(list.Data))
	for _, m := range list.Data {
		if m.ID == "" {
			continue
		}
		models = append(models, modelEntry(m.ID))
	}
	return models, nil
}

// modelEntry builds one provider model entry.
func modelEntry(id string) map[string]any {
	return map[string]any{
		"id":            id,
		"name":          fmt.Sprintf("%s %s", providerName, id),
		"contextWindow": contextWindow,
		"maxTokens":     maxTokens,
		"reasoning":     false,
		"toolCalls":     true,
	}
}

// modelIDs extracts the model IDs from a catalog.
func modelIDs(models []map[string]any) []string {
	ids := make([]string, 0, len(models))
	for _, m := range models {
		if id, _ := m["id"].(string); id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

// fallbackModels is a minimal catalog used only when the public catalog
// endpoint cannot be reached, so the provider still registers.
func fallbackModels() []map[string]any {
	ids := []string{
		"qwen3-coder:480b",
		"gpt-oss:120b",
		"deepseek-v3.1:qwen-3-235b",
		"doubao-seed-1.6:flash",
		"seedoss:36b",
	}
	models := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		models = append(models, modelEntry(id))
	}
	return models
}

// ---------------------------------------------------------------------------
// Quota usage
// ---------------------------------------------------------------------------

// ollamaUsageResponse is the /api/usage payload. The usage values are the
// fraction of each window's quota that has been consumed, in [0, 1).
type ollamaUsageResponse struct {
	Limits struct {
		Session struct {
			Usage float64 `json:"usage"`
		} `json:"session"`
		Weekly struct {
			Usage float64 `json:"usage"`
		} `json:"weekly"`
	} `json:"limits"`
}

// fetchUsage reads the account's consumed quota fractions for the current
// session and week. The key is required.
func fetchUsage(base, key string) (session, weekly float64, err error) {
	var payload ollamaUsageResponse
	if err := getJSON(base+"/api/usage", key, &payload); err != nil {
		return 0, 0, err
	}
	return payload.Limits.Session.Usage, payload.Limits.Weekly.Usage, nil
}

// usageStatusText renders the remaining quota for the footer status entry.
// consumed values are fractions in [0, 1]; values outside the range are
// clamped.
func usageStatusText(session, weekly float64) string {
	remain := func(used float64) int {
		left := clamp01(1 - clamp01(used))
		return int(left * 100)
	}
	return fmt.Sprintf("oc session %d%% · week %d%% left", remain(session), remain(weekly))
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// ---------------------------------------------------------------------------
// Provider registration
// ---------------------------------------------------------------------------

// providerConfig assembles the runtime provider registration. When key is
// empty, the wire config uses environment interpolation so the secret never
// appears in extension source; the host resolves it from the process
// environment, or from the credential stored by `pig login ollama-cloud`.
//
// The config is JSON-serializable and never contains the OAuth closure; the
// login flow is attached once at construction time.
func providerConfig(key string, models []map[string]any) sdk.ProviderConfig {
	if key == "" {
		key = "$" + envAPIKey
	}
	return sdk.ProviderConfig{
		"name":    providerName,
		"baseUrl": baseURL() + "/v1",
		"api":     "openai-completions",
		"apiKey":  key,
		"models":  models,
	}
}

// oauthProvider declares the interactive login flow: prompt for the API key
// and let the host store the result in auth.json. Ollama Cloud has no refresh
// flow; the key is the bearer token.
func oauthProvider() *sdk.OAuthProvider {
	return &sdk.OAuthProvider{
		Name: providerName,
		Login: func(cb *sdk.OAuthLoginCallbacks) (sdk.OAuthCredentials, error) {
			key, err := cb.OnPrompt(sdk.OAuthPrompt{
				Message:     fmt.Sprintf("%s API key (%s)", providerName, settingsURL),
				Placeholder: "sk-...",
			})
			if err != nil {
				return sdk.OAuthCredentials{}, err
			}
			key = strings.TrimSpace(key)
			if key == "" {
				return sdk.OAuthCredentials{}, errors.New("empty API key")
			}
			return sdk.OAuthCredentials{Access: key}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

// usage is the process-lifetime footer updater. The session context is
// refreshed by every session_start; ticks and turn ends re-render the status
// from the latest context and stop when the connection is gone.
type usageUpdater struct {
	mu  sync.Mutex
	ctx sdk.Context

	startOnce sync.Once
}

var updater usageUpdater

// Extension is the conventional PiG factory. Registration runs without a
// credential store, so the initial catalog is the public live list.
func Extension() *sdk.Extension {
	e := sdk.New("pig-ollama-cloud")

	models, err := fetchModels(baseURL(), apiKey())
	if err != nil || len(models) == 0 {
		models = fallbackModels()
	}
	// Attach the interactive login flow only to the construction-time
	// registration; runtime re-registration over the wire cannot serialize
	// the OAuth closures. The flow stays registered until the extension
	// shuts down or is reloaded.
	cfg := providerConfig(apiKey(), models)
	cfg["oauth"] = oauthProvider()
	e.RegisterProvider(providerID, cfg)

	e.Tool("ollama_cloud_status",
		"Check the Ollama Cloud connection: whether an API key is configured, which models are available, and the remaining session and weekly quota.",
		sdk.Schema{"type": "object", "properties": map[string]any{}},
		ollamaCloudStatus)

	e.OnSessionStart(sessionStart)
	e.OnEvent(sdk.EventTurnEnd, func(ctx sdk.Context, _ map[string]any) (any, error) {
		updater.refreshStatus()
		return nil, nil
	})

	return e
}

// ollamaCloudStatus implements the ollama_cloud_status tool. The catalog is
// public, so the model list is reported even without a credential.
func ollamaCloudStatus(ctx sdk.Context, _ map[string]any) (any, error) {
	result := map[string]any{
		"apiKeyConfigured": apiKey() != "",
		"baseUrl":          baseURL(),
	}

	models, err := fetchModels(baseURL(), apiKey())
	if err != nil {
		result["reachable"] = false
		result["error"] = err.Error()
		return result, nil
	}
	result["reachable"] = true
	result["models"] = modelIDs(models)

	key := apiKey()
	if key == "" {
		if resolved := ctx.ModelRegistry().GetApiKeyForProvider(providerID); resolved != nil {
			key = *resolved
		}
	}
	if key != "" {
		if session, weekly, err := fetchUsage(baseURL(), key); err == nil {
			result["sessionUsed"] = session
			result["weeklyUsed"] = weekly
		}
	}
	return result, nil
}

// sessionStart upgrades the registration with the credential resolved the
// same way a model request resolves it (env var, or the key stored by
// `pig login ollama-cloud`). The factory runs before credentials can be
// resolved, so the initial registration may carry an unresolved
// $OLLAMA_CLOUD_API_KEY interpolation; re-registering with the resolved key
// and the live catalog makes the models visible in the selector. It also
// installs the footer usage updater.
func sessionStart(ctx sdk.Context, _ map[string]any) (any, error) {
	resolved := ctx.ModelRegistry().GetApiKeyForProvider(providerID)
	if resolved == nil || *resolved == "" {
		// No credential anywhere: clear the footer entry and leave the
		// initial registration as is.
		ctx.SetStatus(providerID, "")
		return nil, nil
	}

	models, err := fetchModels(baseURL(), *resolved)
	if err == nil && len(models) > 0 {
		_ = ctx.ModelRegistry().RegisterProvider(providerID, providerConfig(*resolved, models))
	}

	updater.mu.Lock()
	updater.ctx = ctx
	updater.mu.Unlock()

	// Render the first quota reading immediately, then keep it fresh. The
	// updater owns one goroutine for the process lifetime; /reload replaces
	// the whole process.
	updater.startOnce.Do(func() { go updater.run() })
	updater.refreshStatus()

	return nil, nil
}

// current returns the latest session context, or a zero Context.
func (u *usageUpdater) current() sdk.Context {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.ctx
}

// ctxValid reports whether ctx carries a live extension connection. Host
// methods on a zero Context would panic on its nil connection.
func ctxValid(ctx sdk.Context) bool {
	return ctx != (sdk.Context{})
}

// run polls Ollama Cloud for quota usage and re-renders the footer entry. It
// stops by process teardown, or earlier when the extension connection dies.
func (u *usageUpdater) run() {
	ticker := time.NewTicker(usageInterval)
	defer ticker.Stop()
	for range ticker.C {
		if u.current().Err() != nil {
			return
		}
		u.refreshStatus()
	}
}

// refreshStatus renders the quota status entry from the latest session
// context. It never blocks its caller long: on the happy path it is one
// usage request; errors keep the previous footer text.
func (u *usageUpdater) refreshStatus() {
	ctx := u.current()
	if !ctxValid(ctx) || ctx.Err() != nil || !ctx.HasUI() {
		return
	}
	resolved := ctx.ModelRegistry().GetApiKeyForProvider(providerID)
	if resolved == nil || *resolved == "" {
		ctx.SetStatus(providerID, "")
		return
	}
	session, weekly, err := fetchUsage(baseURL(), *resolved)
	if err != nil {
		return // keep the previous reading rather than flashing an error
	}
	ctx.SetStatus(providerID, usageStatusText(session, weekly))
}
