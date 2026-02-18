/**
 * Test Setup — In-memory SQLite database with full schema migrations
 *
 * Creates a fresh in-memory database before each test run with all tables
 * from sessions 3 through 18 of the recipe-app evolution.
 */

const Database = require('better-sqlite3');
const crypto = require('crypto');

let db;

/** Run all migrations inline — mirrors the cumulative schema at session 18 */
function initializeDatabase() {
  db = new Database(':memory:');
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');

  // Session 3: core recipes table
  db.exec(`
    CREATE TABLE recipes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      ingredients TEXT NOT NULL,
      instructions TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Session 5: dietary tags
  db.exec(`ALTER TABLE recipes ADD COLUMN dietary_tags TEXT DEFAULT '[]'`);

  // Session 7: image and prep time
  db.exec(`ALTER TABLE recipes ADD COLUMN image_url TEXT DEFAULT ''`);
  db.exec(`ALTER TABLE recipes ADD COLUMN prep_time INTEGER DEFAULT 0`);

  // Session 10: meal plans and structured ingredients
  db.exec(`
    CREATE TABLE meal_plans (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      week_start TEXT NOT NULL,
      name TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.exec(`
    CREATE TABLE meal_plan_items (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      plan_id INTEGER NOT NULL REFERENCES meal_plans(id) ON DELETE CASCADE,
      recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
      day_of_week TEXT NOT NULL,
      meal_type TEXT NOT NULL
    )
  `);

  db.exec(`
    CREATE TABLE recipe_ingredients (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
      name TEXT NOT NULL,
      amount REAL NOT NULL,
      unit TEXT NOT NULL,
      category TEXT DEFAULT 'other'
    )
  `);

  // Session 12: recipe sharing
  db.exec(`
    CREATE TABLE recipe_shares (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
      code TEXT NOT NULL UNIQUE,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Session 18: users and ownership
  db.exec(`
    CREATE TABLE users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT NOT NULL UNIQUE,
      email TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      display_name TEXT DEFAULT '',
      dietary_preferences TEXT DEFAULT '[]',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.exec(`ALTER TABLE recipes ADD COLUMN owner_id INTEGER REFERENCES users(id)`);

  return db;
}

/** Create a test recipe with sensible defaults */
function createTestRecipe(overrides = {}) {
  const data = {
    title: overrides.title || 'Test Pasta',
    ingredients: overrides.ingredients || 'pasta, sauce, cheese',
    instructions: overrides.instructions || 'Boil pasta. Add sauce. Top with cheese.',
    dietary_tags: overrides.dietary_tags || '[]',
    image_url: overrides.image_url || '',
    prep_time: overrides.prep_time || 30,
    owner_id: overrides.owner_id || null,
  };

  const stmt = db.prepare(`
    INSERT INTO recipes (title, ingredients, instructions, dietary_tags, image_url, prep_time, owner_id)
    VALUES (@title, @ingredients, @instructions, @dietary_tags, @image_url, @prep_time, @owner_id)
  `);
  const result = stmt.run(data);
  return { id: result.lastInsertRowid, ...data };
}

/** Create a test user with a hashed password */
function createTestUser(overrides = {}) {
  const password = overrides.password || 'testpassword123';
  const salt = crypto.randomBytes(16).toString('hex');
  const password_hash = crypto.createHash('sha256').update(password + salt).digest('hex') + ':' + salt;

  const data = {
    username: overrides.username || 'testuser',
    email: overrides.email || 'test@example.com',
    password_hash,
    display_name: overrides.display_name || 'Test User',
    dietary_preferences: overrides.dietary_preferences || '[]',
  };

  const stmt = db.prepare(`
    INSERT INTO users (username, email, password_hash, display_name, dietary_preferences)
    VALUES (@username, @email, @password_hash, @display_name, @dietary_preferences)
  `);
  const result = stmt.run(data);
  return { id: result.lastInsertRowid, ...data, password };
}

/** Generate a fake JWT-like token for testing */
function getTestToken(userId, username) {
  const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64url');
  const payload = Buffer.from(JSON.stringify({
    userId,
    username: username || 'testuser',
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 3600,
  })).toString('base64url');
  const signature = crypto.createHmac('sha256', 'test-secret-key')
    .update(`${header}.${payload}`)
    .digest('base64url');
  return `${header}.${payload}.${signature}`;
}

/** Reset database — drop all rows but keep schema */
function resetDatabase() {
  db.exec('DELETE FROM meal_plan_items');
  db.exec('DELETE FROM meal_plans');
  db.exec('DELETE FROM recipe_shares');
  db.exec('DELETE FROM recipe_ingredients');
  db.exec('DELETE FROM recipes');
  db.exec('DELETE FROM users');
  // Reset autoincrement counters
  db.exec("DELETE FROM sqlite_sequence");
}

/** Close database connection */
function closeDatabase() {
  if (db) db.close();
}

/** Get a reference to the current database */
function getDb() {
  return db;
}

module.exports = {
  initializeDatabase,
  resetDatabase,
  closeDatabase,
  getDb,
  createTestRecipe,
  createTestUser,
  getTestToken,
};
