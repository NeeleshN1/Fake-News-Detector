"use client";

import { useState } from "react";
import { labelFromScore } from "@/lib/news";
import RatingBadge from "./RatingBadge";

const MODEL_BASE =
  process.env.NEXT_PUBLIC_API_URL || "https://fake-news-headline-detector-api.onrender.com";

export default function AnalyzeSheet() {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [err, setErr] = useState(false);

  const analyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setErr(false);
    setScore(null);
    try {
      const res = await fetch(`${MODEL_BASE}/predict?headline=${encodeURIComponent(text)}`, {
        method: "POST",
      });
      const data = await res.json();
      setScore(typeof data.confidence === "number" ? data.confidence : null);
    } catch {
      setErr(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Trigger */}
      <button
        onClick={() => setOpen(true)}
        className="absolute left-1/2 top-[max(0.85rem,env(safe-area-inset-top))] z-40 -translate-x-1/2 rounded-full bg-white/10 px-3 py-1.5 text-xs font-bold uppercase tracking-wide text-white ring-1 ring-white/20 backdrop-blur-md transition hover:bg-white/20"
      >
        ✎ Analyze
      </button>

      {!open ? null : (
        <div className="absolute inset-0 z-50 flex items-end justify-center" onClick={() => setOpen(false)}>
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-[520px] rounded-t-3xl border-t border-white/10 bg-neutral-950 p-5 pb-8"
          >
            <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-white/25" />
            <h3 className="font-display text-xl uppercase tracking-tight text-white">Analyze a headline</h3>
            <p className="mb-3 text-xs text-white/50">
              Scored by the C++ DistilBERT model.
            </p>

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
              placeholder="Paste a news headline…"
              className="w-full resize-none rounded-xl border border-white/10 bg-neutral-900 p-3 text-white placeholder-white/30 outline-none focus:border-red-500/50"
            />

            <button
              onClick={analyze}
              disabled={loading}
              className="mt-3 w-full rounded-xl bg-red-600 py-2.5 font-bold text-white transition hover:bg-red-500 disabled:opacity-60"
            >
              {loading ? "Analyzing…" : "Analyze"}
            </button>

            {err && <p className="mt-3 text-sm text-red-400">Couldn&apos;t reach the model API.</p>}

            {score !== null && (
              <div className="mt-4 flex items-center justify-between rounded-xl bg-white/5 p-3">
                <div className="text-sm text-white/70">
                  Fake score: <span className="font-mono text-white">{score.toFixed(3)}</span>
                  <span className="ml-1 text-xs text-white/40">(0 = real, 1 = fake)</span>
                </div>
                <RatingBadge rating={{ label: labelFromScore(score), score }} />
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
