/**
 * Authentication and authorization middleware.
 *
 * This file provides Express middleware that wraps the JWT verification
 * logic using settings from config/auth.js. It is separate from the
 * auth config to keep configuration and runtime behavior in different
 * layers.
 *
 * Introduced: session 18 (user accounts + auth)
 *
 * Usage:
 *   const { requireAuth, requireRole } = require('./middleware/auth');
 *   router.post('/recipes', requireAuth, recipeController.create);
 *   router.delete('/admin/users/:id', requireAuth, requireRole('admin'), adminController.deleteUser);
 */

const jwt = require('jsonwebtoken');
const authConfig = require('../../config/auth');

/**
 * Verify the JWT from the Authorization header and attach the decoded
 * user payload to req.user.
 *
 * Expects: Authorization: Bearer <token>
 */
function requireAuth(req, res, next) {
  const header = req.headers.authorization;
  if (!header || !header.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Authentication required' });
  }

  const token = header.slice(7); // strip "Bearer "

  try {
    const decoded = jwt.verify(token, authConfig.jwt.secret, {
      algorithms: [authConfig.jwt.algorithm],
    });
    req.user = decoded; // { id, email, role, iat, exp }
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired' });
    }
    return res.status(401).json({ error: 'Invalid token' });
  }
}

/**
 * Restrict access to users with a specific role.
 * Must be used after requireAuth.
 *
 * @param {string} role - Required role (e.g. "admin")
 * @returns {Function} Express middleware
 */
function requireRole(role) {
  return (req, res, next) => {
    if (!req.user || req.user.role !== role) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }
    next();
  };
}

// NOTE: requireOwnership() is NOT implemented.
// This is a known gap — any authenticated user can currently update or
// delete any recipe. The fix requires comparing req.user.id against the
// recipe's author_id, but the Recipe model refactor (session 22) hasn't
// landed yet.  See TODO in routes/recipes.js.

module.exports = { requireAuth, requireRole };
