import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "");
    if (!apiBase) return [];
    return [
      { source: "/tv", destination: `${apiBase}/tv` },
      { source: "/tv/:path*", destination: `${apiBase}/tv/:path*` },
      { source: "/tv-api/:path*", destination: `${apiBase}/tv-api/:path*` },
    ];
  },
};

export default nextConfig;
