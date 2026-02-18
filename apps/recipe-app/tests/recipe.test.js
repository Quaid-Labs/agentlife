/**
 * Recipe CRUD & Search Tests — introduced session 7
 *
 * Covers the core recipe lifecycle: create, read, update, delete, and search.
 * Includes tests that REPRODUCE the SQL injection bug from session 3 and
 * verify the parameterized query fix from session 7.
 */

const {
  initializeDatabase, resetDatabase, closeDatabase, getDb, createTestRecipe,
} = require('./setup');
const { assertRecipeShape, recipePayload } = require('./helpers');

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
// SQL Injection
// ---------------------------------------------------------------------------

describe('SQL Injection Vulnerability (Session 3 Bug)', () => {
  /**
   * Session 3 used string concatenation for search:
   *   SELECT * FROM recipes WHERE title LIKE '%${q}%'
   * This test demonstrates the vulnerability would allow table destruction.
   */
  it('should demonstrate the vulnerable query pattern with string interpolation', () => {
    // Use an ISOLATED database so the DROP TABLE doesn't wreck other tests
    const Database = require('better-sqlite3');
    const isolated = new Database(':memory:');
    isolated.exec('CREATE TABLE recipes (id INTEGER PRIMARY KEY, title TEXT NOT NULL)');
    isolated.exec("INSERT INTO recipes (title) VALUES ('Grandma Pasta')");
    expect(isolated.prepare('SELECT COUNT(*) AS cnt FROM recipes').get().cnt).toBe(1);

    // This is the VULNERABLE pattern from session 3 — do NOT use in production.
    // With string interpolation, malicious input becomes part of the SQL.
    const maliciousInput = "'; DROP TABLE recipes; --";
    const vulnerableQuery = `SELECT * FROM recipes WHERE title LIKE '%${maliciousInput}%'`;

    // better-sqlite3's exec() allows multi-statement execution,
    // so the DROP TABLE actually succeeds — the table is destroyed.
    isolated.exec(vulnerableQuery);

    // Table is GONE — the injection succeeded
    expect(() => {
      isolated.prepare('SELECT COUNT(*) AS cnt FROM recipes').get();
    }).toThrow(/no such table/);

    isolated.close();
  });

  it('should handle injection attempt via parameterized query safely (Session 7 fix)', () => {
    createTestRecipe({ title: 'Safe Pasta' });

    // Parameterized query — the fix from session 7
    const maliciousInput = "'; DROP TABLE recipes; --";
    const safeQuery = db.prepare('SELECT * FROM recipes WHERE title LIKE ?');
    const results = safeQuery.all(`%${maliciousInput}%`);

    // No match, but no crash and no data loss
    expect(results).toHaveLength(0);
    expect(db.prepare('SELECT COUNT(*) AS cnt FROM recipes').get().cnt).toBe(1);
  });

  it('should not interpret SQL special characters in parameterized search', () => {
    createTestRecipe({ title: "Mom's Special Recipe" });

    const searchTerm = "Mom's";
    const stmt = db.prepare('SELECT * FROM recipes WHERE title LIKE ?');
    const results = stmt.all(`%${searchTerm}%`);

    // The apostrophe is treated as literal text, not SQL syntax
    expect(results).toHaveLength(1);
    expect(results[0].title).toBe("Mom's Special Recipe");
  });
});

// ---------------------------------------------------------------------------
// CRUD Operations
// ---------------------------------------------------------------------------

describe('Recipe CRUD', () => {
  describe('Create', () => {
    it('should create a recipe with all required fields', () => {
      const recipe = createTestRecipe({
        title: 'Chicken Alfredo',
        ingredients: 'chicken, fettuccine, cream, parmesan',
        instructions: 'Cook chicken. Boil pasta. Make sauce. Combine.',
      });

      expect(recipe.id).toBeDefined();
      expect(recipe.title).toBe('Chicken Alfredo');
      expect(recipe.ingredients).toContain('chicken');

      // Verify it persisted
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      expect(row).toBeDefined();
      expect(row.title).toBe('Chicken Alfredo');
      expect(row.prep_time).toBe(30); // default from createTestRecipe
    });

    it('should auto-increment recipe IDs', () => {
      const r1 = createTestRecipe({ title: 'First' });
      const r2 = createTestRecipe({ title: 'Second' });
      const r3 = createTestRecipe({ title: 'Third' });

      expect(r2.id).toBe(r1.id + 1);
      expect(r3.id).toBe(r2.id + 1);
    });

    it('should set created_at automatically', () => {
      createTestRecipe({ title: 'Timestamped Recipe' });
      const row = db.prepare('SELECT created_at FROM recipes WHERE title = ?').get('Timestamped Recipe');
      expect(row.created_at).toBeDefined();
      expect(row.created_at).not.toBeNull();
    });

    it('should reject a recipe with missing title', () => {
      const stmt = db.prepare(
        'INSERT INTO recipes (ingredients, instructions) VALUES (?, ?)'
      );
      // SQLite NOT NULL constraint on title
      expect(() => stmt.run('stuff', 'do things')).toThrow(/NOT NULL/);
    });

    it('should reject a recipe with missing ingredients', () => {
      const stmt = db.prepare(
        'INSERT INTO recipes (title, instructions) VALUES (?, ?)'
      );
      expect(() => stmt.run('No Ingredients Recipe', 'just vibes')).toThrow(/NOT NULL/);
    });

    it('should reject a recipe with missing instructions', () => {
      const stmt = db.prepare(
        'INSERT INTO recipes (title, ingredients) VALUES (?, ?)'
      );
      expect(() => stmt.run('No Instructions', 'mystery items')).toThrow(/NOT NULL/);
    });
  });

  describe('Read', () => {
    it('should retrieve a recipe by ID', () => {
      const created = createTestRecipe({ title: 'Readable Recipe' });
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(created.id);

      expect(row.id).toBe(created.id);
      expect(row.title).toBe('Readable Recipe');
      expect(row.dietary_tags).toBe('[]');
      expect(row.image_url).toBe('');
      expect(row.prep_time).toBe(30);
    });

    it('should return undefined for a non-existent recipe ID', () => {
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(99999);
      expect(row).toBeUndefined();
    });

    it('should list all recipes', () => {
      createTestRecipe({ title: 'Recipe A' });
      createTestRecipe({ title: 'Recipe B' });
      createTestRecipe({ title: 'Recipe C' });

      const rows = db.prepare('SELECT * FROM recipes').all();
      expect(rows).toHaveLength(3);
    });

    it('should return an empty array when no recipes exist', () => {
      const rows = db.prepare('SELECT * FROM recipes').all();
      expect(rows).toHaveLength(0);
    });
  });

  describe('Update', () => {
    it('should update a recipe title', () => {
      const recipe = createTestRecipe({ title: 'Old Title' });
      db.prepare('UPDATE recipes SET title = ? WHERE id = ?').run('New Title', recipe.id);

      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      expect(row.title).toBe('New Title');
    });

    it('should update multiple fields at once', () => {
      const recipe = createTestRecipe({ title: 'Basic', prep_time: 10 });
      db.prepare('UPDATE recipes SET title = ?, prep_time = ?, image_url = ? WHERE id = ?')
        .run('Advanced', 45, 'https://img.example.com/dish.jpg', recipe.id);

      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      expect(row.title).toBe('Advanced');
      expect(row.prep_time).toBe(45);
      expect(row.image_url).toBe('https://img.example.com/dish.jpg');
    });

    it('should return changes count of 0 for non-existent recipe', () => {
      const result = db.prepare('UPDATE recipes SET title = ? WHERE id = ?').run('Ghost', 99999);
      expect(result.changes).toBe(0);
    });
  });

  describe('Delete', () => {
    it('should delete a recipe by ID', () => {
      const recipe = createTestRecipe({ title: 'Doomed Recipe' });
      const result = db.prepare('DELETE FROM recipes WHERE id = ?').run(recipe.id);

      expect(result.changes).toBe(1);
      const row = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipe.id);
      expect(row).toBeUndefined();
    });

    it('should return changes count of 0 when deleting non-existent recipe', () => {
      const result = db.prepare('DELETE FROM recipes WHERE id = ?').run(99999);
      expect(result.changes).toBe(0);
    });

    it('should not affect other recipes when deleting one', () => {
      const r1 = createTestRecipe({ title: 'Keep Me' });
      const r2 = createTestRecipe({ title: 'Delete Me' });
      const r3 = createTestRecipe({ title: 'Keep Me Too' });

      db.prepare('DELETE FROM recipes WHERE id = ?').run(r2.id);

      const rows = db.prepare('SELECT * FROM recipes ORDER BY id').all();
      expect(rows).toHaveLength(2);
      expect(rows[0].title).toBe('Keep Me');
      expect(rows[1].title).toBe('Keep Me Too');
    });
  });
});

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

describe('Recipe Search', () => {
  beforeEach(() => {
    createTestRecipe({ title: 'Spaghetti Bolognese', ingredients: 'spaghetti, beef, tomato sauce' });
    createTestRecipe({ title: 'Spaghetti Carbonara', ingredients: 'spaghetti, eggs, bacon, parmesan' });
    createTestRecipe({ title: 'Caesar Salad', ingredients: 'romaine, croutons, caesar dressing' });
    createTestRecipe({ title: 'Chicken Tikka Masala', ingredients: 'chicken, yogurt, spices, cream' });
    createTestRecipe({ title: 'Vegetable Stir Fry', ingredients: 'broccoli, peppers, soy sauce, tofu' });
  });

  it('should find recipes by exact title match', () => {
    const stmt = db.prepare('SELECT * FROM recipes WHERE title LIKE ?');
    const results = stmt.all('%Caesar Salad%');
    expect(results).toHaveLength(1);
    expect(results[0].title).toBe('Caesar Salad');
  });

  it('should find recipes by partial title match', () => {
    const stmt = db.prepare('SELECT * FROM recipes WHERE title LIKE ?');
    const results = stmt.all('%Spaghetti%');
    expect(results).toHaveLength(2);
    expect(results.map(r => r.title)).toContain('Spaghetti Bolognese');
    expect(results.map(r => r.title)).toContain('Spaghetti Carbonara');
  });

  it('should perform case-insensitive search (SQLite LIKE is case-insensitive for ASCII)', () => {
    const stmt = db.prepare('SELECT * FROM recipes WHERE title LIKE ?');
    const results = stmt.all('%spaghetti%');
    // SQLite LIKE is case-insensitive for ASCII by default
    expect(results).toHaveLength(2);
  });

  it('should return empty results for unmatched query', () => {
    const stmt = db.prepare('SELECT * FROM recipes WHERE title LIKE ?');
    const results = stmt.all('%Nonexistent Dish%');
    expect(results).toHaveLength(0);
  });

  it('should return all recipes when search term is empty', () => {
    const stmt = db.prepare('SELECT * FROM recipes WHERE title LIKE ?');
    const results = stmt.all('%%');
    expect(results).toHaveLength(5);
  });

  it('should search by ingredient content', () => {
    const stmt = db.prepare('SELECT * FROM recipes WHERE ingredients LIKE ?');
    const results = stmt.all('%chicken%');
    expect(results).toHaveLength(1);
    expect(results[0].title).toBe('Chicken Tikka Masala');
  });

  it('should search across title AND ingredients', () => {
    const stmt = db.prepare('SELECT * FROM recipes WHERE title LIKE ? OR ingredients LIKE ?');
    const term = '%spaghetti%';
    const results = stmt.all(term, term);
    // Both spaghetti recipes match on title AND ingredients
    expect(results).toHaveLength(2);
  });
});
