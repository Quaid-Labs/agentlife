/**
 * Input validation middleware and validator factories.
 *
 * Usage:
 *   const { validate, required, maxLength, isArray, isIn } = require('./validation');
 *
 *   router.post('/recipes',
 *     validate({
 *       title: [required(), maxLength(200)],
 *       ingredients: [required(), isArray({ minLength: 1 })],
 *       dietaryTags: [isArray(), each(isIn(DIETARY_LABELS))],
 *     }),
 *     recipeController.create
 *   );
 *
 * Introduced: session 5 (recipe creation endpoint)
 */

// Allowed dietary labels — shared with the front-end enum
const DIETARY_LABELS = [
  'vegetarian', 'vegan', 'gluten-free', 'dairy-free',
  'nut-free', 'keto', 'paleo', 'low-carb', 'halal', 'kosher',
];

// ---- Sanitization helpers ----

/** Trim whitespace from string values. */
function trimValue(value) {
  return typeof value === 'string' ? value.trim() : value;
}

/** Escape HTML special characters to prevent XSS in stored content. */
function escapeHtml(str) {
  if (typeof str !== 'string') return str;
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ---- Validator factories ----
// Each returns a function (value, fieldName) => errorString | null

function required() {
  return (value, field) => {
    if (value === undefined || value === null || value === '') {
      return `${field} is required`;
    }
    return null;
  };
}

function maxLength(max) {
  return (value, field) => {
    if (typeof value === 'string' && value.length > max) {
      return `${field} must be at most ${max} characters`;
    }
    return null;
  };
}

function minLength(min) {
  return (value, field) => {
    if (typeof value === 'string' && value.length < min) {
      return `${field} must be at least ${min} characters`;
    }
    return null;
  };
}

function isArray(opts = {}) {
  const { minLength: minLen } = opts;
  return (value, field) => {
    if (value !== undefined && value !== null && !Array.isArray(value)) {
      return `${field} must be an array`;
    }
    if (minLen && Array.isArray(value) && value.length < minLen) {
      return `${field} must have at least ${minLen} item(s)`;
    }
    return null;
  };
}

function isIn(allowedValues) {
  return (value, field) => {
    if (value !== undefined && value !== null && !allowedValues.includes(value)) {
      return `${field} must be one of: ${allowedValues.join(', ')}`;
    }
    return null;
  };
}

/** Apply a validator to every element of an array field. */
function each(validator) {
  return (value, field) => {
    if (!Array.isArray(value)) return null; // isArray() handles this
    for (let i = 0; i < value.length; i++) {
      const error = validator(value[i], `${field}[${i}]`);
      if (error) return error;
    }
    return null;
  };
}

// ---- Core middleware ----

/**
 * Build an Express middleware that validates req.body against a rules object.
 *
 * @param {Object} rules - Map of field name to array of validator functions.
 * @returns {Function} Express middleware
 */
function validate(rules) {
  return (req, res, next) => {
    const errors = [];

    for (const [field, validators] of Object.entries(rules)) {
      // Sanitize before validation — trim strings in-place
      if (typeof req.body[field] === 'string') {
        req.body[field] = trimValue(req.body[field]);
      }

      const value = req.body[field];
      for (const validator of validators) {
        const error = validator(value, field);
        if (error) {
          errors.push(error);
          break; // one error per field is enough
        }
      }
    }

    if (errors.length) {
      return res.status(400).json({ errors });
    }
    next();
  };
}

module.exports = {
  validate,
  required,
  maxLength,
  minLength,
  isArray,
  isIn,
  each,
  trimValue,
  escapeHtml,
  DIETARY_LABELS,
};
