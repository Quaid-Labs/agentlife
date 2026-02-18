/**
 * Request/response logging middleware.
 *
 * Logs every completed HTTP request with method, URL, status code,
 * response time, and content length. Output is colorized by status
 * code range when writing to a TTY.
 *
 * Introduced: session 10 (observability improvements)
 *
 * Usage:
 *   const { requestLogger } = require('./middleware/logging');
 *   app.use(requestLogger);
 */

const fs = require('fs');
const path = require('path');

// ANSI color codes for terminal output
const COLORS = {
  green: '\x1b[32m',   // 2xx success
  cyan: '\x1b[36m',    // 3xx redirect
  yellow: '\x1b[33m',  // 4xx client error
  red: '\x1b[31m',     // 5xx server error
  reset: '\x1b[0m',
};

/**
 * Pick a color based on HTTP status code.
 * @param {number} status
 * @returns {string} ANSI escape code
 */
function colorForStatus(status) {
  if (status < 300) return COLORS.green;
  if (status < 400) return COLORS.cyan;
  if (status < 500) return COLORS.yellow;
  return COLORS.red;
}

/**
 * Express middleware that logs request completion.
 *
 * Attaches a `finish` listener to the response so the log line
 * includes the final status code and timing.
 */
function requestLogger(req, res, next) {
  const start = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - start;
    const contentLength = res.get('Content-Length') || '-';
    const status = res.statusCode;
    const color = colorForStatus(status);

    const line = `${req.method} ${req.originalUrl || req.url} ${status} ${duration}ms ${contentLength}`;

    // Colorize only when stdout is a TTY
    if (process.stdout.isTTY) {
      console.log(`${color}${line}${COLORS.reset}`);
    } else {
      console.log(line);
    }
  });

  next();
}

module.exports = { requestLogger };
