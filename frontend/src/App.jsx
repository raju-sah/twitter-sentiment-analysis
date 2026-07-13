import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "/api";

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function analyze() {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Request failed");
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const pct = result ? Math.round(result.probability * 100) : 0;

  return (
    <div className="app">
      <h1>Twitter Sentiment Analyzer</h1>
      <p className="subtitle">
        Logistic-regression model trained from scratch on the NLTK Twitter corpus.
      </p>

      <div className="card">
        <textarea
          placeholder="Type a tweet or sentence to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button onClick={analyze} disabled={loading || !text.trim()}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <span className={`badge ${result.label}`}>
              {result.label.toUpperCase()}
            </span>
            <div className="bar">
              <span
                className={result.label}
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="meta">
              Positive probability: <strong>{pct}%</strong>
            </p>
            {result.processed_tokens?.length > 0 && (
              <div className="tokens">
                {result.processed_tokens.map((t, i) => (
                  <span key={i}>{t}</span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
