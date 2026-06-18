// Shared types for the live-news feed.

export type RatingLabel = "REAL" | "FAKE" | "UNCERTAIN";

export interface Rating {
  label: RatingLabel;
  score: number; // P(fake), 0..1 (the model's calibrated fake-likelihood)
}

export interface Article {
  id: string;
  title: string;
  description: string | null;
  source: string;
  url: string;
  image: string | null;
  publishedAt: string | null;
  rating: Rating | null;
}

export interface NewsResponse {
  articles: Article[];
  category: string;
  page: number;
  totalResults: number;
  error?: string;
}

export interface CategoryDef {
  id: string; // NewsAPI category
  label: string; // display label
}

// The 5 bottom-bar categories (NewsAPI top-headlines categories).
export const CATEGORIES: CategoryDef[] = [
  { id: "general", label: "Top" },
  { id: "business", label: "Business" },
  { id: "technology", label: "Tech" },
  { id: "sports", label: "Sports" },
  { id: "health", label: "Health" },
];
