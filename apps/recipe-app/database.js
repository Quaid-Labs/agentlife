const Database = require('better-sqlite3');
const path = require('path');

const db = new Database(path.join(__dirname, 'recipes.db'));

db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
  CREATE TABLE IF NOT EXISTS recipes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    ingredients TEXT NOT NULL,
    instructions TEXT NOT NULL,
    dietary_tags TEXT DEFAULT '[]',
    image_url TEXT DEFAULT '',
    prep_time INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

db.exec(`
  CREATE TABLE IF NOT EXISTS meal_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

db.exec(`
  CREATE TABLE IF NOT EXISTS meal_plan_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES meal_plans(id) ON DELETE CASCADE,
    recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    day_of_week TEXT NOT NULL,
    meal_type TEXT NOT NULL
  )
`);

db.exec(`
  CREATE TABLE IF NOT EXISTS recipe_ingredients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    amount REAL NOT NULL,
    unit TEXT NOT NULL,
    category TEXT DEFAULT 'other'
  )
`);

// Allowed dietary labels — used by front-end and validation
const DIETARY_LABELS = [
  'vegetarian', 'vegan', 'gluten-free', 'dairy-free',
  'nut-free', 'diabetic-friendly', 'low-sodium', 'low-carb',
  'keto', 'paleo',
];

// Preset: recipes safe for Mom (Linda) — diabetic-friendly AND low-sodium
const SAFE_FOR_MOM = ['diabetic-friendly', 'low-sodium'];

// Safe column addition for schema migration
function addColumnIfMissing(table, column, type) {
  const cols = db.pragma(`table_info(${table})`);
  if (!cols.find(c => c.name === column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${type}`);
  }
}

// Recipe sharing (introduced session 12 — GraphQL pivot)
db.exec(`
  CREATE TABLE IF NOT EXISTS recipe_shares (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    code TEXT NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// Users table (introduced session 18 — auth)
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT DEFAULT '',
    dietary_preferences TEXT DEFAULT '[]',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

addColumnIfMissing('recipes', 'dietary_tags', "TEXT DEFAULT '[]'");
addColumnIfMissing('recipes', 'image_url', "TEXT DEFAULT ''");
addColumnIfMissing('recipes', 'prep_time', "INTEGER DEFAULT 0");
addColumnIfMissing('recipes', 'owner_id', "INTEGER REFERENCES users(id)");

module.exports = db;
module.exports.DIETARY_LABELS = DIETARY_LABELS;
module.exports.SAFE_FOR_MOM = SAFE_FOR_MOM;
