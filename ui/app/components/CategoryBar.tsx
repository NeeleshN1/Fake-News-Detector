"use client";

import { CATEGORIES } from "@/lib/types";

// Minimal inline icons keyed by category id.
const ICONS: Record<string, React.ReactNode> = {
  general: (
    <path d="M4 5h16M4 10h16M4 15h10M4 20h7" strokeWidth="2" strokeLinecap="round" />
  ),
  business: (
    <path d="M4 19V11M9 19V5M14 19v-6M19 19V8" strokeWidth="2" strokeLinecap="round" />
  ),
  technology: (
    <>
      <rect x="4" y="6" width="16" height="11" rx="1.5" strokeWidth="2" />
      <path d="M2 21h20" strokeWidth="2" strokeLinecap="round" />
    </>
  ),
  sports: (
    <>
      <circle cx="12" cy="12" r="8" strokeWidth="2" />
      <path d="M12 4v16M4 12h16M6 6l12 12M18 6L6 18" strokeWidth="1.3" />
    </>
  ),
  health: (
    <path
      d="M12 20s-7-4.5-7-9a4 4 0 017-2.6A4 4 0 0119 11c0 4.5-7 9-7 9z"
      strokeWidth="2"
      strokeLinejoin="round"
    />
  ),
};

export default function CategoryBar({
  active,
  onChange,
}: {
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <nav
      className="pointer-events-auto absolute inset-x-0 bottom-0 z-30 flex items-stretch justify-around gap-1 px-2 pb-[max(0.5rem,env(safe-area-inset-bottom))] pt-2"
      style={{
        background: "linear-gradient(to top, rgba(0,0,0,0.92), rgba(0,0,0,0.55) 60%, transparent)",
      }}
    >
      {CATEGORIES.map((c) => {
        const on = c.id === active;
        return (
          <button
            key={c.id}
            onClick={() => onChange(c.id)}
            aria-pressed={on}
            className={`flex flex-1 flex-col items-center gap-1 rounded-lg py-1.5 transition ${
              on ? "text-white" : "text-white/45 hover:text-white/70"
            }`}
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              className={`h-6 w-6 transition-transform ${on ? "scale-110" : ""}`}
            >
              {ICONS[c.id]}
            </svg>
            <span className={`text-[10px] font-bold uppercase tracking-wide ${on ? "" : "font-semibold"}`}>
              {c.label}
            </span>
            <span className={`h-0.5 w-5 rounded-full transition ${on ? "bg-red-500" : "bg-transparent"}`} />
          </button>
        );
      })}
    </nav>
  );
}
