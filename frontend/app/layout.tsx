import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionRAG",
  description: "Multimodal RAG with Gemini Vision — text, figures, and tables",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
