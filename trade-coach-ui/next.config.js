/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: { root: __dirname },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Allow production builds to succeed even if there are TS errors
    ignoreBuildErrors: true,
  },
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "");
    if (!apiBase) return [];
    return [
      { source: "/tv-api/:path*", destination: `${apiBase}/tv-api/:path*` },
    ];
  },
};

module.exports = nextConfig;
