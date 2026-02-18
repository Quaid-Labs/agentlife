const Database = require('better-sqlite3');
const path = require('path');

const DB_PATH = process.env.DATABASE_PATH
  || path.join(__dirname, '..', 'recipes.db');

function createConnection(dbPath) {
  const db = new Database(dbPath || DB_PATH);
  db.pragma('journal_mode = WAL');
  db.pragma('busy_timeout = 5000');
  return db;
}

module.exports = { createConnection, DB_PATH };
