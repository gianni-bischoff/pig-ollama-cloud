package pig_ollama_cloud

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestBaseURL(t *testing.T) {
	t.Setenv(envBaseURL, "")
	if got := baseURL(); got != defaultBase {
		t.Errorf("baseURL() with empty env = %q, want %q", got, defaultBase)
	}

	t.Setenv(envBaseURL, "https://cloud.example.com/")
	if got := baseURL(); got != "https://cloud.example.com" {
		t.Errorf("baseURL() trailing slash = %q, want https://cloud.example.com", got)
	}

	t.Setenv(envBaseURL, "cloud.example.com")
	if got := baseURL(); got != "https://cloud.example.com" {
		t.Errorf("baseURL() without scheme = %q, want https://cloud.example.com", got)
	}
}

func TestAPIKeyPrecedence(t *testing.T) {
	t.Setenv(envAPIKey, " primary ")
	t.Setenv(envAPIAlt, "alt")
	if got := apiKey(); got != "primary" {
		t.Errorf("apiKey() = %q, want trimmed primary key", got)
	}

	t.Setenv(envAPIKey, "")
	if got := apiKey(); got != "alt" {
		t.Errorf("apiKey() fallback = %q, want alt", got)
	}

	t.Setenv(envAPIAlt, "  ")
	if got := apiKey(); got != "" {
		t.Errorf("apiKey() whitespace-only = %q, want empty", got)
	}
}

func TestGetJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "bad auth", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	var out map[string]any
	if err := getJSON(server.URL, "test-key", &out); err != nil {
		t.Fatalf("getJSON: %v", err)
	}
	if out["ok"] != true {
		t.Errorf("getJSON decoded = %v, want {ok: true}", out)
	}
	if err := getJSON(server.URL, "wrong-key", &out); err == nil {
		t.Error("getJSON with bad auth succeeded, want error")
	}
}

func TestFetchModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"id":"qwen3-coder:480b"},{"id":"gpt-oss:120b"},{"id":""}]}`))
	}))
	defer server.Close()

	// The catalog is public: fetching without a key must succeed.
	models, err := fetchModels(server.URL, "")
	if err != nil {
		t.Fatalf("fetchModels without key: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("fetchModels returned %d models, want 2 (empty id skipped)", len(models))
	}
	if models[0]["id"] != "qwen3-coder:480b" {
		t.Errorf("first model id = %v, want qwen3-coder:480b", models[0]["id"])
	}
	if models[0]["contextWindow"] != contextWindow {
		t.Errorf("contextWindow = %v, want %d", models[0]["contextWindow"], contextWindow)
	}
}

func TestFetchUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/usage" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"activity":{"cost":"0.00"},"limits":{"session":{"usage":0.616},"weekly":{"usage":0.617}}}`))
	}))
	defer server.Close()

	session, weekly, err := fetchUsage(server.URL, "key")
	if err != nil {
		t.Fatalf("fetchUsage: %v", err)
	}
	if session != 0.616 || weekly != 0.617 {
		t.Errorf("fetchUsage = (%v, %v), want (0.616, 0.617)", session, weekly)
	}
}

func TestUsageStatusText(t *testing.T) {
	tests := []struct {
		session, weekly float64
		want            string
	}{
		{0.616, 0.617, "oc session 38% · week 38% left"},
		{0, 0, "oc session 100% · week 100% left"},
		{1, 1, "oc session 0% · week 0% left"},
		{1.5, -0.2, "oc session 0% · week 100% left"}, // clamped
	}
	for _, tc := range tests {
		if got := usageStatusText(tc.session, tc.weekly); got != tc.want {
			t.Errorf("usageStatusText(%v, %v) = %q, want %q", tc.session, tc.weekly, got, tc.want)
		}
	}
}

func TestFallbackModels(t *testing.T) {
	models := fallbackModels()
	if len(models) == 0 {
		t.Fatal("fallbackModels returned no models")
	}
	ids := modelIDs(models)
	if len(ids) != len(models) {
		t.Errorf("modelIDs dropped entries: %d of %d", len(ids), len(models))
	}
}

func TestProviderConfig(t *testing.T) {
	t.Setenv(envBaseURL, "")

	cfg := providerConfig("", fallbackModels())
	if got := cfg["apiKey"]; got != "$OLLAMA_CLOUD_API_KEY" {
		t.Errorf("providerConfig empty key: apiKey = %v, want env interpolation", got)
	}

	cfg = providerConfig("literal-key", fallbackModels())
	if got := cfg["apiKey"]; got != "literal-key" {
		t.Errorf("providerConfig literal key: apiKey = %v, want literal-key", got)
	}
	if got := cfg["baseUrl"]; got != defaultBase+"/v1" {
		t.Errorf("baseUrl = %v, want %s/v1", got, defaultBase)
	}
	if cfg["api"] != "openai-completions" {
		t.Errorf("api = %v, want openai-completions", cfg["api"])
	}
	// The runtime config must stay JSON-serializable: no OAuth closure.
	if _, err := json.Marshal(cfg); err != nil {
		t.Errorf("providerConfig is not JSON-serializable: %v", err)
	}
}
