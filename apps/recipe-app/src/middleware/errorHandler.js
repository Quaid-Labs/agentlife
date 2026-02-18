/**
 * Centralized error handling middleware and custom error class.
 *
 * All thrown/next(err) errors funnel through errorHandler(). In
 * development, full stack traces are returned; in production, only
 * operational error messages are exposed to the client.
 *
 * Introduced: session 7 (error handling overhaul)
 *
 * Usage:
 *   const { AppError } = require('./errorHandler');
 *   throw new AppError('Recipe not found', 404);
 *
 *   // At the bottom of app.js:
 *   const { errorHandler, notFoundHandler } = require('./errorHandler');
 *   app.use(notFoundHandler);
 *   app.use(errorHandler);
 */

/**
 * Custom application error with HTTP status code.
 * Operational errors (user input, not-found, auth) are safe to expose.
 * Programming errors (bugs, null refs) are NOT operational and get
 * a generic 500 response in production.
 */
class AppError extends Error {
  /**
   * @param {string} message - Human-readable error message
   * @param {number} [statusCode=500] - HTTP status code
   */
  constructor(message, statusCode = 500) {
    super(message);
    this.name = 'AppError';
    this.statusCode = statusCode;
    this.isOperational = true;

    // Capture stack trace without this constructor in it
    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * Format a timestamp for error logs.
 * @returns {string} ISO timestamp
 */
function timestamp() {
  return new Date().toISOString();
}

/**
 * Express error-handling middleware (4 arguments).
 */
function errorHandler(err, req, res, _next) {
  const statusCode = err.statusCode || 500;
  const isProduction = process.env.NODE_ENV === 'production';

  // Always log the error server-side
  console.error(`[${timestamp()}] ${req.method} ${req.url} -> ${statusCode}: ${err.message}`);
  if (!isProduction && err.stack) {
    console.error(err.stack);
  }

  // Build client response
  const response = {
    error: err.isOperational ? err.message : 'Internal server error',
    statusCode,
  };

  // In development, include the stack trace for debugging
  if (!isProduction) {
    response.stack = err.stack;
  }

  res.status(statusCode).json(response);
}

/**
 * Catch-all 404 handler — mount before errorHandler.
 */
function notFoundHandler(req, res, _next) {
  res.status(404).json({
    error: `Cannot ${req.method} ${req.url}`,
    statusCode: 404,
  });
}

module.exports = {
  AppError,
  errorHandler,
  notFoundHandler,
};
