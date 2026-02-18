/**
 * Simple in-memory rate limiter middleware.
 *
 * Tracks request counts per IP address within a sliding time window.
 * No external dependencies (no Redis) — suitable for single-process
 * deployments. For multi-instance setups, swap in a Redis-backed
 * rate limiter.
 *
 * Introduced: session 16 (API abuse prevention)
 *
 * Usage:
 *   const { rateLimiter } = require('./middleware/rateLimiter');
 *   app.use('/api', rateLimiter({ windowMs: 60000, max: 30 }));
 */

/**
 * Create a rate-limiting middleware.
 *
 * @param {Object} [options]
 * @param {number} [options.windowMs=900000] - Window duration in ms (default 15 min)
 * @param {number} [options.max=100]         - Max requests per window per IP
 * @returns {Function} Express middleware
 */
function rateLimiter({ windowMs = 15 * 60 * 1000, max = 100 } = {}) {
  // Map<ip, { count, resetTime }>
  const hits = new Map();

  // Periodically purge expired entries to prevent unbounded memory growth.
  // The interval is unref'd so it doesn't keep the process alive.
  const cleanup = setInterval(() => {
    const now = Date.now();
    for (const [ip, entry] of hits) {
      if (now >= entry.resetTime) {
        hits.delete(ip);
      }
    }
  }, windowMs);
  cleanup.unref();

  return (req, res, next) => {
    const ip = req.ip || req.socket.remoteAddress;
    const now = Date.now();

    let entry = hits.get(ip);

    // First request or window expired — start fresh
    if (!entry || now >= entry.resetTime) {
      entry = { count: 1, resetTime: now + windowMs };
      hits.set(ip, entry);
      return next();
    }

    entry.count += 1;

    if (entry.count > max) {
      const retryAfterSec = Math.ceil((entry.resetTime - now) / 1000);
      res.set('Retry-After', String(retryAfterSec));
      return res.status(429).json({
        error: 'Too many requests, please try again later',
        retryAfterSeconds: retryAfterSec,
      });
    }

    next();
  };
}

module.exports = { rateLimiter };
