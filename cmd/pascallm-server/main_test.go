package main

import "testing"

// TestCheckLoopback は、ループバックの IP アドレスだけを待ち受けに使えることを確かめます。
func TestCheckLoopback(t *testing.T) {
	cases := []struct {
		addr string
		ok   bool
	}{
		{"127.0.0.1:18120", true},
		{"127.0.0.2:18120", true},
		{"[::1]:18120", true},
		{"0.0.0.0:18120", false},
		{":18120", false},
		{"localhost:18120", false},
		{"192.0.2.10:18120", false},
		{"100.64.0.10:18120", false},
		{"127.0.0.1", false},
	}
	for _, c := range cases {
		if err := checkLoopback(c.addr); (err == nil) != c.ok {
			t.Errorf("checkLoopback(%q) error = %v, want ok = %v", c.addr, err, c.ok)
		}
	}
}

// TestArchitectureDefaults は Gemma の既定値と明示したモデル名・思考設定を検証します。
func TestArchitectureDefaults(t *testing.T) {
	opts := options{architecture: "gemma4"}
	if err := resolveOptions(&opts); err != nil {
		t.Fatal(err)
	}
	if opts.pq4Path != "./data/gemma4-12b-qat.pq4m" || opts.enableThinking || opts.modelName != "gemma4-12b-qat-pq4" {
		t.Fatal(opts)
	}
	explicit := options{architecture: "gemma4", pq4Path: "/tmp/custom.pq4m", modelName: "custom", thinkingSet: true, enableThinking: true}
	if err := resolveOptions(&explicit); err != nil {
		t.Fatal(err)
	}
	if explicit.pq4Path != "/tmp/custom.pq4m" || explicit.modelName != "custom" || !explicit.enableThinking {
		t.Fatal(explicit)
	}
	invalid := options{architecture: "unknown"}
	if resolveOptions(&invalid) == nil {
		t.Fatal("unknown architecture accepted")
	}
}
