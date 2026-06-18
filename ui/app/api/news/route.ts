// Server-side news proxy. Keeps the NewsAPI key secret (its free plan blocks browser
// CORS), cleans titles, batch-scores every headline with the fake-news model, and returns
// merged Article[] to the client. Cached ~5 min to respect NewsAPI's 100 req/day limit.

import { NextRequest, NextResponse } from "next/server";
import { cleanTitle, labelFromScore } from "@/lib/news";
import type { Article, NewsResponse } from "@/lib/types";

export const revalidate = 300;

const NEWSAPI = "https://newsapi.org/v2/top-headlines";
const KEY = process.env.NEWSAPI_KEY;
const MODEL_BASE =
  process.env.MODEL_API_BASE || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const PAGE_SIZE = 20;

interface RawArticle {
  title: string | null;
  description: string | null;
  url: string;
  urlToImage: string | null;
  publishedAt: string | null;
  source?: { name?: string };
}

export async function GET(req: NextRequest) {
  const category = req.nextUrl.searchParams.get("category") || "general";
  const page = Math.max(1, Number(req.nextUrl.searchParams.get("page") || "1"));

  if (!KEY) {
    return NextResponse.json(
      { articles: [], category, page, totalResults: 0, error: "NEWSAPI_KEY not set" } as NewsResponse,
      { status: 500 },
    );
  }

  // 1) Fetch live headlines (cached).
  let raw: { status?: string; message?: string; totalResults?: number; articles?: RawArticle[] };
  try {
    const url =
      `${NEWSAPI}?country=us&category=${encodeURIComponent(category)}` +
      `&pageSize=${PAGE_SIZE}&page=${page}&apiKey=${KEY}`;
    const r = await fetch(url, { next: { revalidate } });
    raw = await r.json();
    if (raw.status !== "ok") throw new Error(raw.message || "NewsAPI error");
  } catch (e) {
    return NextResponse.json(
      { articles: [], category, page, totalResults: 0, error: String(e) } as NewsResponse,
      { status: 200 },
    );
  }

  // 2) Normalize + clean.
  const articles: Article[] = (raw.articles || [])
    .filter((a) => a.title && a.title !== "[Removed]")
    .map((a) => ({
      id: a.url || `${category}-${page}-${a.title}`,
      title: cleanTitle(a.title as string),
      description: a.description ?? null,
      source: a.source?.name ?? "Unknown",
      url: a.url,
      image: a.urlToImage ?? null,
      publishedAt: a.publishedAt ?? null,
      rating: null,
    }));

  // 3) Batch-score the headlines with the model (best-effort: degrade to no rating).
  if (articles.length) {
    try {
      const res = await fetch(`${MODEL_BASE}/predict_batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ headlines: articles.map((a) => a.title) }),
        cache: "no-store",
      });
      if (res.ok) {
        const ratings: { label: string; confidence: number }[] = await res.json();
        articles.forEach((a, i) => {
          const r = ratings[i];
          if (r && typeof r.confidence === "number") {
            a.rating = { label: labelFromScore(r.confidence), score: r.confidence };
          }
        });
      }
    } catch {
      // model offline -> articles render without a badge
    }
  }

  const body: NewsResponse = {
    articles,
    category,
    page,
    totalResults: raw.totalResults ?? articles.length,
  };
  return NextResponse.json(body);
}
