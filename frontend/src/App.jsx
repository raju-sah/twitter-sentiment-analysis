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

      <footer className="footer">
        <a href="https://sahraju.com.np" target="_blank" rel="noopener noreferrer">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
          sahraju.com.np
        </a>
        <a href="mailto:try.rajusah@gmail.com">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
          try.rajusah@gmail.com
        </a>
        <a href="https://linkedin.com/in/rajusah18" target="_blank" rel="noopener noreferrer">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
          linkedin.com/in/rajusah18
        </a>
        <a href="https://github.com/raju-sah" target="_blank" rel="noopener noreferrer">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
          github.com/raju-sah
        </a>
        <a href="https://instagram.com/okay.raju" target="_blank" rel="noopener noreferrer">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/></svg>
          instagram.com/okay.raju
        </a>
      </footer>
    </div>
  );
}
