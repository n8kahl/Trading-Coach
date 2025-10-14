import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-coach-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-coach-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Trading Coach · Live Plan Console",
  description: "Real-time coaching with Fancy Trader 2.0.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-neutral-950 text-neutral-50 antialiased`}>
        <div className="relative flex min-h-screen flex-col">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(74,222,128,0.15),transparent_55%)]" />
          <main className="relative z-10 flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
