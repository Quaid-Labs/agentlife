/**
 * API configuration — server, rate limits, external services.
 *
 * All values can be overridden via environment variables. Defaults
 * are tuned for local development; production should set them
 * explicitly through .env or container environment.
 *
 * Introduced: session 12 (API hardening pass)
 */

module.exports = {
  // ---- Server ----
  port: parseInt(process.env.PORT, 10) || 3000,

  // ---- CORS ----
  cors: {
    origins: process.env.CORS_ORIGINS
      ? process.env.CORS_ORIGINS.split(',').map((o) => o.trim())
      : ['http://localhost:3000'],
  },

  // ---- Rate limiting (applied globally) ----
  rateLimit: {
    windowMs: 15 * 60 * 1000, // 15-minute sliding window
    max: 100,                  // requests per window per IP
  },

  // ---- Pagination defaults for list endpoints ----
  pagination: {
    defaultLimit: 20,
    maxLimit: 100,
  },

  // ---- Nutrition API (Edamam) ----
  // Used by the /api/recipes/:id/nutrition endpoint (session 14).
  // Leave blank to disable nutrition lookups gracefully.
  nutrition: {
    provider: 'edamam',
    baseUrl: 'https://api.edamam.com/api/nutrition-data',
    appId: process.env.EDAMAM_APP_ID || '',
    appKey: process.env.EDAMAM_APP_KEY || '',
  },
};
