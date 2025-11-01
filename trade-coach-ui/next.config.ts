import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  experimental: {
    externalDir: true,
  },
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "");
    if (!apiBase) return [];
    return [
      { source: "/tv-api/:path*", destination: `${apiBase}/tv-api/:path*` },
    ];
  },
};

export default nextConfig;
