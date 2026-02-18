/**
 * Authentication and session configuration.
 *
 * JWT is the primary auth mechanism. Passwords are hashed with
 * PBKDF2 (Node's built-in crypto) — no bcrypt native dependency.
 *
 * IMPORTANT: The default JWT secret is intentionally insecure.
 * Production deployments MUST set JWT_SECRET via environment.
 *
 * Introduced: session 18 (user accounts + auth)
 */

module.exports = {
  // ---- JSON Web Token ----
  jwt: {
    secret: process.env.JWT_SECRET || 'recipe-app-dev-secret-change-me',
    expiresIn: '24h',
    algorithm: 'HS256',
  },

  // ---- Password hashing (PBKDF2) ----
  password: {
    iterations: 10000,
    keyLength: 64,    // bytes — produces a 128-char hex digest
    digest: 'sha512',
    saltLength: 32,   // bytes of random salt per password
  },

  // ---- Session limits ----
  session: {
    maxAge: 24 * 60 * 60 * 1000, // 24 hours in milliseconds
  },
};
