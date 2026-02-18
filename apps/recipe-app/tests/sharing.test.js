/**
 * Recipe Sharing Tests — introduced session 12, expanded session 16
 *
 * Tests the share code generation system: creating unique codes,
 * retrieving recipes by code, and edge cases around re-sharing.
 */

const {
  initializeDatabase, resetDatabase, closeDatabase, getDb, createTestRecipe,
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
// Helpers: mirror app sharing logic
// ---------------------------------------------------------------------------

function shareRecipe(recipeId) {
  const recipe = db.prepare('SELECT * FROM recipes WHERE id = ?').get(recipeId);
  if (!recipe) return null;

  // Check if already shared
  const existing = db.prepare('SELECT * FROM recipe_shares WHERE recipe_id = ?').get(recipeId);
  if (existing) return existing;

  // Generate a unique code (8 chars alphanumeric)
  const code = Math.random().toString(36).substring(2, 10);
  db.prepare('INSERT INTO recipe_shares (recipe_id, code) VALUES (?, ?)').run(recipeId, code);
  return db.prepare('SELECT * FROM recipe_shares WHERE recipe_id = ?').get(recipeId);
}

function getRecipeByShareCode(code) {
  const share = db.prepare('SELECT * FROM recipe_shares WHERE code = ?').get(code);
  if (!share) return null;
  return db.prepare('SELECT * FROM recipes WHERE id = ?').get(share.recipe_id) || null;
}

// ---------------------------------------------------------------------------
// Share Code Generation
// ---------------------------------------------------------------------------

describe('Share Code Generation', () => {
  it('should generate a share code for a recipe', () => {
    const recipe = createTestRecipe({ title: 'Shareable Pasta' });
    const share = shareRecipe(recipe.id);

    expect(share).not.toBeNull();
    expect(share.code).toBeDefined();
    expect(typeof share.code).toBe('string');
    expect(share.code.length).toBeGreaterThanOrEqual(4);
    expect(share.recipe_id).toBe(recipe.id);
  });

  it('should set created_at timestamp on share', () => {
    const recipe = createTestRecipe({ title: 'Timestamped Share' });
    const share = shareRecipe(recipe.id);

    expect(share.created_at).toBeDefined();
    expect(share.created_at).not.toBeNull();
  });

  it('should generate unique codes for different recipes', () => {
    const r1 = createTestRecipe({ title: 'Recipe One' });
    const r2 = createTestRecipe({ title: 'Recipe Two' });
    const r3 = createTestRecipe({ title: 'Recipe Three' });

    const s1 = shareRecipe(r1.id);
    const s2 = shareRecipe(r2.id);
    const s3 = shareRecipe(r3.id);

    // All codes should be different
    const codes = [s1.code, s2.code, s3.code];
    const uniqueCodes = new Set(codes);
    expect(uniqueCodes.size).toBe(3);
  });

  it('should return null when sharing a non-existent recipe', () => {
    const result = shareRecipe(99999);
    expect(result).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Re-sharing (Idempotency)
// ---------------------------------------------------------------------------

describe('Re-sharing Same Recipe', () => {
  it('should return the same share code when sharing the same recipe again', () => {
    const recipe = createTestRecipe({ title: 'Idempotent Recipe' });

    const first = shareRecipe(recipe.id);
    const second = shareRecipe(recipe.id);

    expect(first.code).toBe(second.code);
    expect(first.id).toBe(second.id);
  });

  it('should not create duplicate entries in recipe_shares', () => {
    const recipe = createTestRecipe({ title: 'No Duplicates' });

    shareRecipe(recipe.id);
    shareRecipe(recipe.id);
    shareRecipe(recipe.id);

    const shares = db.prepare('SELECT * FROM recipe_shares WHERE recipe_id = ?').all(recipe.id);
    expect(shares).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Retrieve by Share Code
// ---------------------------------------------------------------------------

describe('Retrieve Recipe by Share Code', () => {
  it('should retrieve the correct recipe by share code', () => {
    const recipe = createTestRecipe({
      title: 'Retrievable Recipe',
      ingredients: 'flour, eggs, milk',
      instructions: 'Mix and bake.',
      prep_time: 25,
    });
    const share = shareRecipe(recipe.id);
    const retrieved = getRecipeByShareCode(share.code);

    expect(retrieved).not.toBeNull();
    expect(retrieved.id).toBe(recipe.id);
    expect(retrieved.title).toBe('Retrievable Recipe');
    expect(retrieved.ingredients).toBe('flour, eggs, milk');
    expect(retrieved.prep_time).toBe(25);
  });

  it('should return null for an invalid share code', () => {
    const result = getRecipeByShareCode('nonexistent');
    expect(result).toBeNull();
  });

  it('should return null for an empty share code', () => {
    const result = getRecipeByShareCode('');
    expect(result).toBeNull();
  });

  it('should return the full recipe including dietary tags', () => {
    const recipe = createTestRecipe({
      title: 'Tagged Shared Recipe',
      dietary_tags: JSON.stringify(['vegan', 'gluten-free']),
    });
    const share = shareRecipe(recipe.id);
    const retrieved = getRecipeByShareCode(share.code);

    expect(retrieved.dietary_tags).toBe(JSON.stringify(['vegan', 'gluten-free']));
    const tags = JSON.parse(retrieved.dietary_tags);
    expect(tags).toContain('vegan');
    expect(tags).toContain('gluten-free');
  });

  it('should return the full recipe including image_url and prep_time', () => {
    const recipe = createTestRecipe({
      title: 'Full Shared Recipe',
      image_url: 'https://img.example.com/pasta.jpg',
      prep_time: 45,
    });
    const share = shareRecipe(recipe.id);
    const retrieved = getRecipeByShareCode(share.code);

    expect(retrieved.image_url).toBe('https://img.example.com/pasta.jpg');
    expect(retrieved.prep_time).toBe(45);
  });
});

// ---------------------------------------------------------------------------
// Edge Cases
// ---------------------------------------------------------------------------

describe('Sharing Edge Cases', () => {
  it('should handle sharing a recipe that is then deleted', () => {
    const recipe = createTestRecipe({ title: 'Doomed Shared Recipe' });
    const share = shareRecipe(recipe.id);
    const code = share.code;

    // Delete the recipe — CASCADE should delete the share too
    db.prepare('DELETE FROM recipes WHERE id = ?').run(recipe.id);

    // Share should be gone due to CASCADE
    const shareRow = db.prepare('SELECT * FROM recipe_shares WHERE code = ?').get(code);
    expect(shareRow).toBeUndefined();

    // Retrieval by code should fail
    const retrieved = getRecipeByShareCode(code);
    expect(retrieved).toBeNull();
  });

  it('should enforce unique constraint on share codes', () => {
    const recipe = createTestRecipe({ title: 'Unique Code Test' });

    // Manually insert a share with a known code
    db.prepare('INSERT INTO recipe_shares (recipe_id, code) VALUES (?, ?)').run(recipe.id, 'fixedcode');

    // Trying to insert another share with the same code should fail
    const recipe2 = createTestRecipe({ title: 'Another Recipe' });
    expect(() => {
      db.prepare('INSERT INTO recipe_shares (recipe_id, code) VALUES (?, ?)').run(recipe2.id, 'fixedcode');
    }).toThrow(/UNIQUE/);
  });

  it('should handle multiple recipes each with their own share code', () => {
    const recipes = [];
    for (let i = 0; i < 10; i++) {
      const r = createTestRecipe({ title: `Batch Recipe ${i}` });
      recipes.push(r);
    }

    const shares = recipes.map(r => shareRecipe(r.id));

    // Each share should work independently
    shares.forEach((share, i) => {
      const retrieved = getRecipeByShareCode(share.code);
      expect(retrieved.title).toBe(`Batch Recipe ${i}`);
    });

    // All codes should be unique
    const codes = shares.map(s => s.code);
    expect(new Set(codes).size).toBe(10);
  });
});
