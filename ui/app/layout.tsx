import type { Metadata, Viewport } from "next";
import { Archivo_Black, Inter } from "next/font/google";
import "./globals.css";

const display = Archivo_Black({
  weight: "400",
  variable: "--font-archivo",
  subsets: ["latin"],
});

const sans = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VERIFY — Live News, Fact-Rated",
  description:
    "A TikTok-style live news feed where every headline is rated real or fake by a C++ DistilBERT model.",
};

export const viewport: Viewport = {
  themeColor: "#000000",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${sans.variable} bg-black antialiased`}>{children}</body>
    </html>
  );
}
