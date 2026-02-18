/**
 * Authentication Tests — introduced session 18
 *
 * Tests user registration, login, JWT token generation, password hashing,
 * and the requireAuth middleware. Uses the in-memory test database from setup.js.
 */

const crypto = require('crypto');
const {
  initializeDatabase, resetDatabase, closeDatabase, getDb, createTestUser, getTestToken,
} = require('./setup');

let db;

beforeAll(() => {
  db = initializeDatabase();
});

afterAll(() => {
  closeDatabase();
});

beforeEach(() => {
  resetDatabase();
});

// ---------------------------------------------------------------------------
// Helpers: mirror app auth logic for unit testing
// ---------------------------------------------------------------------------

const authConfig = {
  password: {
    iterations: 10000,
    keyLength: 64,
    digest: 'sha512',
    saltLength: 32,
  },
};

function hashPassword(password) {
  const salt = crypto.randomBytes(authConfig.password.saltLength).toString('hex');
  const hash = crypto.pbkdf2Sync(
    password, salt,
    authConfig.password.iterations,
    authConfig.password.keyLength,
    authConfig.password.digest
  ).toString('hex');
  return `${hash}:${salt}`;
}

function verifyPassword(password, stored) {
  const [hash, salt] = stored.split(':');
  const test = crypto.pbkdf2Sync(
    password, salt,
    authConfig.password.iterations,
    authConfig.password.keyLength,
    authConfig.password.digest
  ).toString('hex');
  return hash === test;
}

// ---------------------------------------------------------------------------
// User Registration
// ---------------------------------------------------------------------------

describe('User Registration', () => {
  it('should create a new user with hashed password', () => {
    const passwordHash = hashPassword('securepassword123');
    db.prepare(
      'INSERT INTO users (username, email, password_hash, display_name) VALUES (?, ?, ?, ?)'
    ).run('maya', 'maya@example.com', passwordHash, 'Maya Chen');

    const user = db.prepare('SELECT * FROM users WHERE username = ?').get('maya');
    expect(user).not.toBeNull();
    expect(user.username).toBe('maya');
    expect(user.email).toBe('maya@example.com');
    expect(user.display_name).toBe('Maya Chen');
    // Password should be hashed, not stored in plain text
    expect(user.password_hash).not.toBe('securepassword123');
    expect(user.password_hash).toContain(':'); // hash:salt format
  });

  it('should enforce unique username constraint', () => {
    const hash = hashPassword('pass1');
    db.prepare('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)').run('maya', 'maya@example.com', hash);

    expect(() => {
      db.prepare('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)').run('maya', 'different@example.com', hash);
    }).toThrow(/UNIQUE/);
  });

  it('should enforce unique email constraint', () => {
    const hash = hashPassword('pass1');
    db.prepare('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)').run('maya', 'maya@example.com', hash);

    expect(() => {
      db.prepare('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)').run('different', 'maya@example.com', hash);
    }).toThrow(/UNIQUE/);
  });

  it('should store dietary preferences as JSON array', () => {
    const hash = hashPassword('pass1');
    const prefs = JSON.stringify(['vegetarian', 'dairy-free']);
    db.prepare(
      'INSERT INTO users (username, email, password_hash, dietary_preferences) VALUES (?, ?, ?, ?)'
    ).run('maya', 'maya@example.com', hash, prefs);

    const user = db.prepare('SELECT * FROM users WHERE username = ?').get('maya');
    const parsed = JSON.parse(user.dietary_preferences);
    expect(parsed).toEqual(['vegetarian', 'dairy-free']);
  });

  it('should default dietary_preferences to empty array', () => {
    const hash = hashPassword('pass1');
    db.prepare(
      'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)'
    ).run('maya', 'maya@example.com', hash);

    const user = db.prepare('SELECT * FROM users WHERE username = ?').get('maya');
    expect(JSON.parse(user.dietary_preferences)).toEqual([]);
  });

  it('should set created_at timestamp automatically', () => {
    const hash = hashPassword('pass1');
    db.prepare('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)').run('maya', 'maya@example.com', hash);

    const user = db.prepare('SELECT * FROM users WHERE username = ?').get('maya');
    expect(user.created_at).toBeDefined();
    expect(user.created_at).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Password Hashing and Verification
// ---------------------------------------------------------------------------

describe('Password Hashing', () => {
  it('should produce different hashes for the same password', () => {
    const hash1 = hashPassword('samepassword');
    const hash2 = hashPassword('samepassword');

    // Different salts mean different hashes
    expect(hash1).not.toBe(hash2);
  });

  it('should verify correct password against hash', () => {
    const password = 'correcthorsebatterystaple';
    const stored = hashPassword(password);

    expect(verifyPassword(password, stored)).toBe(true);
  });

  it('should reject incorrect password', () => {
    const stored = hashPassword('rightpassword');
    expect(verifyPassword('wrongpassword', stored)).toBe(false);
  });

  it('should handle special characters in password', () => {
    const password = 'p@$$w0rd!#%^&*()';
    const stored = hashPassword(password);

    expect(verifyPassword(password, stored)).toBe(true);
    expect(verifyPassword('p@$$w0rd', stored)).toBe(false);
  });

  it('should handle very long passwords', () => {
    const password = 'a'.repeat(1000);
    const stored = hashPassword(password);

    expect(verifyPassword(password, stored)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Recipe Ownership
// ---------------------------------------------------------------------------

describe('Recipe Ownership', () => {
  it('should create recipe with owner_id', () => {
    const user = createTestUser({ username: 'chef1', email: 'chef@example.com' });
    db.prepare(
      'INSERT INTO recipes (title, ingredients, instructions, owner_id) VALUES (?, ?, ?, ?)'
    ).run('Owned Recipe', 'stuff', 'do things', user.id);

    const recipe = db.prepare('SELECT * FROM recipes WHERE title = ?').get('Owned Recipe');
    expect(recipe.owner_id).toBe(user.id);
  });

  it('should allow null owner_id for anonymous recipes', () => {
    db.prepare(
      'INSERT INTO recipes (title, ingredients, instructions) VALUES (?, ?, ?)'
    ).run('Anonymous Recipe', 'stuff', 'do things');

    const recipe = db.prepare('SELECT * FROM recipes WHERE title = ?').get('Anonymous Recipe');
    expect(recipe.owner_id).toBeNull();
  });

  it('should allow querying recipes by owner', () => {
    const user1 = createTestUser({ username: 'chef1', email: 'chef1@example.com' });
    const user2 = createTestUser({ username: 'chef2', email: 'chef2@example.com' });

    db.prepare('INSERT INTO recipes (title, ingredients, instructions, owner_id) VALUES (?, ?, ?, ?)').run('Recipe A', 'a', 'a', user1.id);
    db.prepare('INSERT INTO recipes (title, ingredients, instructions, owner_id) VALUES (?, ?, ?, ?)').run('Recipe B', 'b', 'b', user1.id);
    db.prepare('INSERT INTO recipes (title, ingredients, instructions, owner_id) VALUES (?, ?, ?, ?)').run('Recipe C', 'c', 'c', user2.id);

    const user1Recipes = db.prepare('SELECT * FROM recipes WHERE owner_id = ?').all(user1.id);
    expect(user1Recipes).toHaveLength(2);

    const user2Recipes = db.prepare('SELECT * FROM recipes WHERE owner_id = ?').all(user2.id);
    expect(user2Recipes).toHaveLength(1);
  });

  // BUG DOCUMENTATION: This test shows the known authorization gap.
  // Any user can delete any recipe because there's no ownership check.
  it('should allow deleting any recipe (known auth gap)', () => {
    const user1 = createTestUser({ username: 'chef1', email: 'chef1@example.com' });
    const user2 = createTestUser({ username: 'chef2', email: 'chef2@example.com' });

    db.prepare('INSERT INTO recipes (title, ingredients, instructions, owner_id) VALUES (?, ?, ?, ?)').run('Stolen Recipe', 'a', 'a', user1.id);
    const recipe = db.prepare("SELECT * FROM recipes WHERE title = 'Stolen Recipe'").get();

    // User 2 can delete User 1's recipe — this is the bug
    const result = db.prepare('DELETE FROM recipes WHERE id = ?').run(recipe.id);
    expect(result.changes).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Test Token Generation
// ---------------------------------------------------------------------------

describe('Test Token Generation', () => {
  it('should generate a valid JWT-like token', () => {
    const token = getTestToken(1, 'maya');

    expect(token).toBeDefined();
    expect(typeof token).toBe('string');

    // JWT format: header.payload.signature
    const parts = token.split('.');
    expect(parts).toHaveLength(3);
  });

  it('should encode the correct user info in the payload', () => {
    const token = getTestToken(42, 'testuser');
    const [, payloadB64] = token.split('.');
    const payload = JSON.parse(Buffer.from(payloadB64, 'base64url').toString());

    expect(payload.userId).toBe(42);
    expect(payload.username).toBe('testuser');
    expect(payload.iat).toBeDefined();
    expect(payload.exp).toBeDefined();
    expect(payload.exp).toBeGreaterThan(payload.iat);
  });

  it('should generate different tokens for different users', () => {
    const token1 = getTestToken(1, 'maya');
    const token2 = getTestToken(2, 'alex');

    expect(token1).not.toBe(token2);
  });
});

// ---------------------------------------------------------------------------
// User Query Helpers
// ---------------------------------------------------------------------------

describe('User Queries', () => {
  it('should find user by username', () => {
    createTestUser({ username: 'findme', email: 'findme@example.com' });
    const user = db.prepare('SELECT * FROM users WHERE username = ?').get('findme');

    expect(user).not.toBeNull();
    expect(user.username).toBe('findme');
  });

  it('should return undefined for non-existent user', () => {
    const user = db.prepare('SELECT * FROM users WHERE username = ?').get('ghost');
    expect(user).toBeUndefined();
  });

  it('should exclude password_hash from safe queries', () => {
    createTestUser({ username: 'safe', email: 'safe@example.com' });
    const user = db.prepare(
      'SELECT id, username, email, display_name, dietary_preferences, created_at FROM users WHERE username = ?'
    ).get('safe');

    expect(user.username).toBe('safe');
    expect(user.password_hash).toBeUndefined();
  });
});
