// Real/Fake rating pill shown in the top-right of each news card.
import type { Rating } from "@/lib/types";

const STYLES: Record<string, { label: string; text: string; bg: string; bar: string; ring: string }> = {
  REAL: {
    label: "REAL",
    text: "text-emerald-300",
    bg: "bg-emerald-500/15",
    bar: "bg-emerald-400",
    ring: "ring-emerald-400/40",
  },
  FAKE: {
    label: "FAKE",
    text: "text-red-300",
    bg: "bg-red-500/15",
    bar: "bg-red-400",
    ring: "ring-red-400/40",
  },
  UNCERTAIN: {
    label: "UNSURE",
    text: "text-amber-300",
    bg: "bg-amber-500/15",
    bar: "bg-amber-400",
    ring: "ring-amber-400/40",
  },
};

export default function RatingBadge({ rating }: { rating: Rating | null }) {
  if (!rating) return null;

  const s = STYLES[rating.label] ?? STYLES.UNCERTAIN;
  // confidence in the shown verdict (P(real)=1-score for REAL, P(fake)=score for FAKE)
  const conf =
    rating.label === "REAL"
      ? 1 - rating.score
      : rating.label === "FAKE"
        ? rating.score
        : Math.max(rating.score, 1 - rating.score);
  const pct = Math.round(conf * 100);

  return (
    <div
      className={`pointer-events-none flex flex-col items-end gap-1 rounded-xl ${s.bg} px-2.5 py-1.5 ring-1 ${s.ring} backdrop-blur-md`}
    >
      <div className="flex items-center gap-1.5">
        <span className={`text-[11px] font-extrabold uppercase tracking-wider ${s.text}`}>{s.label}</span>
        <span className="font-display text-base leading-none text-white">{pct}%</span>
      </div>
      <div className="h-1 w-16 overflow-hidden rounded-full bg-white/15">
        <div className={`h-full ${s.bar}`} style={{ width: `${Math.max(6, pct)}%` }} />
      </div>
    </div>
  );
}
