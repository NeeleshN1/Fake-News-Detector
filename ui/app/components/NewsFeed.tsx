"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Article, NewsResponse } from "@/lib/types";
import { CATEGORIES } from "@/lib/types";
import NewsCard from "./NewsCard";
import CategoryBar from "./CategoryBar";
import AnalyzeSheet from "./AnalyzeSheet";

const PAGE_SIZE = 20;

export default function NewsFeed() {
  const [category, setCategory] = useState(CATEGORIES[0].id);
  const [articles, setArticles] = useState<Article[]>([]);
  const [active, setActive] = useState(0);
  const [status, setStatus] = useState<"loading" | "ready" | "error" | "empty">("loading");

  const scroller = useRef<HTMLDivElement>(null);
  const loading = useRef(false);
  const page = useRef(1);
  const exhausted = useRef(false); // NewsAPI ran out -> rotate existing
  const total = useRef(0);
  const rotation = useRef(0);

  const fetchPage = useCallback(async (cat: string, p: number): Promise<NewsResponse | null> => {
    try {
      const res = await fetch(`/api/news?category=${cat}&page=${p}`, { cache: "no-store" });
      if (!res.ok) return null;
      return (await res.json()) as NewsResponse;
    } catch {
      return null;
    }
  }, []);

  // Load (or reload) a category from page 1.
  const loadCategory = useCallback(
    async (cat: string) => {
      loading.current = true;
      setStatus("loading");
      setArticles([]);
      page.current = 1;
      exhausted.current = false;
      rotation.current = 0;
      scroller.current?.scrollTo({ top: 0 });

      const data = await fetchPage(cat, 1);
      loading.current = false;
      if (!data || data.error) {
        setStatus("error");
        return;
      }
      total.current = data.totalResults;
      if (data.articles.length === 0) {
        setStatus("empty");
        return;
      }
      exhausted.current = data.articles.length >= data.totalResults || data.articles.length < PAGE_SIZE;
      setArticles(data.articles);
      setActive(0);
      setStatus("ready");
    },
    [fetchPage],
  );

  useEffect(() => {
    loadCategory(category);
  }, [category, loadCategory]);

  // Append the next page, or rotate the existing articles for infinite scroll.
  const loadMore = useCallback(async () => {
    if (loading.current) return;
    loading.current = true;

    if (!exhausted.current) {
      const next = page.current + 1;
      const data = await fetchPage(category, next);
      if (data && data.articles.length > 0) {
        page.current = next;
        setArticles((prev) => {
          const seen = new Set(prev.map((a) => a.id));
          const fresh = data.articles.filter((a) => !seen.has(a.id));
          if (fresh.length === 0) exhausted.current = true;
          return [...prev, ...fresh];
        });
        if (data.articles.length < PAGE_SIZE) exhausted.current = true;
        loading.current = false;
        return;
      }
      exhausted.current = true;
    }

    // Rotate: re-append the loaded articles with fresh keys (true infinite scroll).
    rotation.current += 1;
    const r = rotation.current;
    setArticles((prev) => {
      const base = prev.slice(0, total.current || prev.length);
      const looped = base.map((a) => ({ ...a, id: `${a.id}~r${r}` }));
      return [...prev, ...looped].slice(-120); // cap memory
    });
    loading.current = false;
  }, [category, fetchPage]);

  const onScroll = useCallback(() => {
    const el = scroller.current;
    if (!el) return;
    const idx = Math.round(el.scrollTop / el.clientHeight);
    if (idx !== active) setActive(idx);
    if (idx >= articles.length - 3) loadMore();
  }, [active, articles.length, loadMore]);

  // Keyboard navigation (desktop).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = scroller.current;
      if (!el) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        el.scrollBy({ top: (e.key === "ArrowDown" ? 1 : -1) * el.clientHeight, behavior: "smooth" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="relative mx-auto h-[100dvh] w-full max-w-[520px] overflow-hidden bg-black sm:border-x sm:border-white/10">
      <div
        ref={scroller}
        onScroll={onScroll}
        className="no-scrollbar h-full snap-y snap-mandatory overflow-y-scroll overscroll-y-contain"
      >
        {status === "ready" &&
          articles.map((a, i) => <NewsCard key={a.id} article={a} active={i === active} />)}

        {status === "loading" && <SkeletonSlide />}

        {status === "error" && (
          <MessageSlide
            title="Couldn't load the feed"
            body="The news service or model API didn't respond. Check that both are running."
            onRetry={() => loadCategory(category)}
          />
        )}
        {status === "empty" && (
          <MessageSlide title="No stories right now" body="Try another category." />
        )}
      </div>

      <CategoryBar active={category} onChange={(id) => id !== category && setCategory(id)} />
      <AnalyzeSheet />
    </div>
  );
}

function SkeletonSlide() {
  return (
    <section className="relative h-[100dvh] w-full snap-start overflow-hidden bg-neutral-950">
      <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-neutral-800 to-neutral-950" />
      <div className="absolute inset-x-0 bottom-0 space-y-3 p-4 pb-28">
        <div className="h-3 w-24 animate-pulse rounded bg-white/20" />
        <div className="h-8 w-5/6 animate-pulse rounded bg-white/20" />
        <div className="h-8 w-2/3 animate-pulse rounded bg-white/20" />
        <div className="h-3 w-full animate-pulse rounded bg-white/10" />
      </div>
    </section>
  );
}

function MessageSlide({
  title,
  body,
  onRetry,
}: {
  title: string;
  body: string;
  onRetry?: () => void;
}) {
  return (
    <section className="relative flex h-[100dvh] w-full snap-start flex-col items-center justify-center gap-3 bg-neutral-950 px-8 text-center">
      <h2 className="font-display text-2xl uppercase text-white">{title}</h2>
      <p className="max-w-xs text-sm text-white/60">{body}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 rounded-full bg-red-600 px-5 py-2 text-sm font-bold text-white hover:bg-red-500"
        >
          Retry
        </button>
      )}
    </section>
  );
}
