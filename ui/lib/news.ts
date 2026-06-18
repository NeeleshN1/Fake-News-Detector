// Helpers for normalizing NewsAPI data (shared by the server route).

import type { RatingLabel } from "./types";

// NewsAPI appends the source to titles, e.g. "Fed holds rates steady - Reuters"
// (sometimes with "|"). Strip that trailing segment for cleaner display + scoring.
export function cleanTitle(title: string): string {
  if (!title) return title;
  const m = title.match(/^(.*\S)\s+[-|–—]\s+[^-|–—]{2,40}$/);
  return (m ? m[1] : title).trim();
}

// Map P(fake) -> a label using the same thresholds as the C++ classifier.
export function labelFromScore(score: number): RatingLabel {
  if (score >= 0.6) return "FAKE";
  if (score <= 0.4) return "REAL";
  return "UNCERTAIN";
}

// Compact relative time, e.g. "2h", "5m", "3d".
export function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const s = Math.max(0, (Date.now() - then) / 1000);
  if (s < 60) return "now";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}
