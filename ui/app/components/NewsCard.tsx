"use client";

import { useState } from "react";
import type { Article } from "@/lib/types";
import { timeAgo } from "@/lib/news";
import RatingBadge from "./RatingBadge";

export default function NewsCard({ article, active }: { article: Article; active: boolean }) {
  const [imgOk, setImgOk] = useState(true);
  const [saved, setSaved] = useState(false);
  const [info, setInfo] = useState(false);

  const share = async () => {
    try {
      if (navigator.share) await navigator.share({ title: article.title, url: article.url });
      else {
        await navigator.clipboard.writeText(article.url);
      }
    } catch {
      /* user cancelled */
    }
  };

  return (
    <section className="relative h-[100dvh] w-full snap-start overflow-hidden bg-neutral-950">
      {/* Background image / fallback */}
      {article.image && imgOk ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={article.image}
          alt=""
          loading="lazy"
          onError={() => setImgOk(false)}
          className="absolute inset-0 h-full w-full object-cover"
        />
      ) : (
        <div className="absolute inset-0 bg-gradient-to-br from-neutral-800 via-neutral-900 to-black" />
      )}

      {/* Scrims for legibility */}
      <div className="absolute inset-0 bg-gradient-to-t from-black via-black/55 to-black/25" />
      <div className="absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-black/70 to-transparent" />

      {/* Top row: brand + rating badge */}
      <div className="absolute inset-x-0 top-0 z-20 flex items-start justify-between p-4 pt-[max(1rem,env(safe-area-inset-top))]">
        <div className="flex items-center gap-2">
          <span className="font-display text-lg tracking-tight text-white">VERIFY</span>
          <span className="flex items-center gap-1 rounded-full bg-red-600 px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wider text-white">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" /> Live
          </span>
        </div>
        <button onClick={() => setInfo((v) => !v)} className="text-left">
          <RatingBadge rating={article.rating} />
        </button>
      </div>

      {/* Rating explanation popover */}
      {info && article.rating && (
        <div className="absolute right-4 top-20 z-30 max-w-[15rem] rounded-xl bg-black/85 p-3 text-xs leading-relaxed text-white/80 ring-1 ring-white/15 backdrop-blur">
          Our DistilBERT model rates the headline&apos;s <b>linguistic style</b> (fake/clickbait vs.
          measured reporting) — a signal, not a fact-check. Score = P(fake) {article.rating.score.toFixed(2)}.
        </div>
      )}

      {/* Right action rail */}
      <div className="absolute bottom-28 right-3 z-20 flex flex-col items-center gap-5 text-white">
        <button onClick={() => setSaved((v) => !v)} className="flex flex-col items-center gap-1">
          <svg viewBox="0 0 24 24" className="h-8 w-8" fill={saved ? "#f43f5e" : "none"} stroke="currentColor">
            <path
              d="M12 20s-7-4.5-7-9a4 4 0 017-2.6A4 4 0 0119 11c0 4.5-7 9-7 9z"
              strokeWidth="2"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-[10px] font-semibold">Like</span>
        </button>
        <button onClick={share} className="flex flex-col items-center gap-1">
          <svg viewBox="0 0 24 24" className="h-8 w-8" fill="none" stroke="currentColor">
            <path d="M4 12v7a1 1 0 001 1h14a1 1 0 001-1v-7M16 6l-4-4-4 4M12 2v14" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-[10px] font-semibold">Share</span>
        </button>
        <a href={article.url} target="_blank" rel="noopener noreferrer" className="flex flex-col items-center gap-1">
          <svg viewBox="0 0 24 24" className="h-8 w-8" fill="none" stroke="currentColor">
            <path d="M14 4h6v6M20 4l-9 9M19 14v5a1 1 0 01-1 1H6a1 1 0 01-1-1V7a1 1 0 011-1h5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-[10px] font-semibold">Read</span>
        </a>
      </div>

      {/* Bottom content */}
      <div
        className={`absolute inset-x-0 bottom-0 z-10 px-4 pb-24 pr-20 transition-all duration-500 ${
          active ? "translate-y-0 opacity-100" : "translate-y-4 opacity-90"
        }`}
      >
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-white/70">
          <span className="rounded bg-white/15 px-2 py-0.5 uppercase tracking-wide text-white">{article.source}</span>
          {article.publishedAt && <span>· {timeAgo(article.publishedAt)} ago</span>}
        </div>
        <h2 className="font-display text-[clamp(1.6rem,6vw,2.3rem)] uppercase leading-[1.03] tracking-tight text-white [text-shadow:0_2px_12px_rgba(0,0,0,0.6)] line-clamp-4">
          {article.title}
        </h2>
        {article.description && (
          <p className="mt-2 line-clamp-2 text-sm text-white/75">{article.description}</p>
        )}
      </div>
    </section>
  );
}
