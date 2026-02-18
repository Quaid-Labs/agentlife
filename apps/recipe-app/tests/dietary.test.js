/**
 * Dietary Tag & Filtering Tests — introduced session 5, expanded session 10
 *
 * Tests the dietary tagging system including filtering by single/multiple tags,
 * the SAFE_FOR_MOM preset, and the DIETARY_LABELS constant.
 */

const {
  initializeDatabase, resetDatabase, closeDatabase, getDb, createTestRecipe,
} = require('./setup');
const { DIETARY_LABELS, SAFE_FOR_MOM } = require('./helpers');

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
// Helper: filter recipes by dietary tags (mirrors app logic)
// ---------------------------------------------------------------------------

/**
 * Filter recipes that contain ALL of the given dietary tags.
 * The app stores dietary_tags as a JSON array string, e.g. '["vegan","gluten-free"]'.
 * Filtering uses JSON parsing in JS (not SQL), matching the app's implementation.
 */
function filterByDietaryTags(recipes, requiredTags) {
  return recipes.filter(recipe => {
    const tags = JSON.parse(recipe.dietary_tags || '[]');
    return requiredTags.every(tag => tags.includes(tag));
  });
}

/** Fetch all recipes from the database */
function getAllRecipes() {
  return db.prepare('SELECT * FROM recipes').all();
}

// ---------------------------------------------------------------------------
// Dietary Labels Constant
// ---------------------------------------------------------------------------

describe('Dietary Labels', () => {
  it('should have exactly 10 dietary labels defined', () => {
    expect(DIETARY_LABELS).toHaveLength(10);
  });

  it('should include all expected labels', () => {
    const expected = [
      'vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'nut-free',
      'diabetic-friendly', 'low-sodium', 'low-carb', 'keto', 'paleo',
    ];
    expected.forEach(label => {
      expect(DIETARY_LABELS).toContain(label);
    });
  });

  it('should not contain duplicate labels', () => {
    const unique = new Set(DIETARY_LABELS);
    expect(unique.size).toBe(DIETARY_LABELS.length);
  });
});

// ---------------------------------------------------------------------------
// SAFE_FOR_MOM Preset
// ---------------------------------------------------------------------------

describe('SAFE_FOR_MOM Preset', () => {
  it('should be defined as diabetic-friendly and low-sodium', () => {
    expect(SAFE_FOR_MOM).toEqual(['diabetic-friendly', 'low-sodium']);
  });

  it('should be a subset of DIETARY_LABELS', () => {
    SAFE_FOR_MOM.forEach(tag => {
      expect(DIETARY_LABELS).toContain(tag);
    });
  });

  it('should filter recipes safe for Mom', () => {
    // Recipe that has BOTH tags — should match
    createTestRecipe({
      title: 'Mom-Safe Chicken Soup',
      dietary_tags: JSON.stringify(['diabetic-friendly', 'low-sodium', 'gluten-free']),
    });
    // Recipe with only one of the two tags — should NOT match
    createTestRecipe({
      title: 'Low Sodium Rice',
      dietary_tags: JSON.stringify(['low-sodium']),
    });
    // Recipe with only the other tag
    createTestRecipe({
      title: 'Diabetic Friendly Cake',
      dietary_tags: JSON.stringify(['diabetic-friendly']),
    });
    // Recipe with neither tag
    createTestRecipe({
      title: 'Regular Burger',
      dietary_tags: JSON.stringify(['keto']),
    });

    const allRecipes = getAllRecipes();
    const safeMom = filterByDietaryTags(allRecipes, SAFE_FOR_MOM);

    expect(safeMom).toHaveLength(1);
    expect(safeMom[0].title).toBe('Mom-Safe Chicken Soup');
  });

  it('should return multiple recipes when several match the preset', () => {
    createTestRecipe({
      title: 'Safe Soup',
      dietary_tags: JSON.stringify(['diabetic-friendly', 'low-sodium']),
    });
    createTestRecipe({
      title: 'Safe Salad',
      dietary_tags: JSON.stringify(['diabetic-friendly', 'low-sodium', 'vegan']),
    });
    createTestRecipe({
      title: 'Unsafe Pizza',
      dietary_tags: JSON.stringify(['vegetarian']),
    });

    const safeMom = filterByDietaryTags(getAllRecipes(), SAFE_FOR_MOM);
    expect(safeMom).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Single Tag Filtering
// ---------------------------------------------------------------------------

describe('Filter by Single Dietary Tag', () => {
  beforeEach(() => {
    createTestRecipe({
      title: 'Vegan Bowl',
      dietary_tags: JSON.stringify(['vegan', 'gluten-free']),
    });
    createTestRecipe({
      title: 'Veggie Burger',
      dietary_tags: JSON.stringify(['vegetarian']),
    });
    createTestRecipe({
      title: 'Keto Steak',
      dietary_tags: JSON.stringify(['keto', 'gluten-free', 'dairy-free']),
    });
    createTestRecipe({
      title: 'Plain Chicken',
      dietary_tags: JSON.stringify([]),
    });
  });

  it('should find recipes with a specific tag', () => {
    const results = filterByDietaryTags(getAllRecipes(), ['vegan']);
    expect(results).toHaveLength(1);
    expect(results[0].title).toBe('Vegan Bowl');
  });

  it('should find recipes with gluten-free tag across multiple recipes', () => {
    const results = filterByDietaryTags(getAllRecipes(), ['gluten-free']);
    expect(results).toHaveLength(2);
    const titles = results.map(r => r.title);
    expect(titles).toContain('Vegan Bowl');
    expect(titles).toContain('Keto Steak');
  });

  it('should return empty array for a tag no recipe has', () => {
    const results = filterByDietaryTags(getAllRecipes(), ['paleo']);
    expect(results).toHaveLength(0);
  });

  it('should include recipes with the tag among other tags', () => {
    const results = filterByDietaryTags(getAllRecipes(), ['dairy-free']);
    expect(results).toHaveLength(1);
    expect(results[0].title).toBe('Keto Steak');
    // Verify the recipe also has other tags
    const tags = JSON.parse(results[0].dietary_tags);
    expect(tags).toContain('keto');
    expect(tags).toContain('gluten-free');
  });
});

// ---------------------------------------------------------------------------
// Multiple Tag Filtering (Intersection)
// ---------------------------------------------------------------------------

describe('Filter by Multiple Dietary Tags (Intersection)', () => {
  beforeEach(() => {
    createTestRecipe({
      title: 'Full Restriction Meal',
      dietary_tags: JSON.stringify(['vegan', 'gluten-free', 'nut-free', 'low-sodium']),
    });
    createTestRecipe({
      title: 'Partial Match',
      dietary_tags: JSON.stringify(['vegan', 'gluten-free']),
    });
    createTestRecipe({
      title: 'Vegan Only',
      dietary_tags: JSON.stringify(['vegan']),
    });
  });

  it('should require ALL tags to match (intersection, not union)', () => {
    const results = filterByDietaryTags(getAllRecipes(), ['vegan', 'gluten-free', 'nut-free']);
    expect(results).toHaveLength(1);
    expect(results[0].title).toBe('Full Restriction Meal');
  });

  it('should return recipes matching two tags', () => {
    const results = filterByDietaryTags(getAllRecipes(), ['vegan', 'gluten-free']);
    expect(results).toHaveLength(2);
    const titles = results.map(r => r.title);
    expect(titles).toContain('Full Restriction Meal');
    expect(titles).toContain('Partial Match');
  });

  it('should return empty when no recipe has all requested tags', () => {
    const results = filterByDietaryTags(getAllRecipes(), ['vegan', 'keto']);
    expect(results).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Edge Cases
// ---------------------------------------------------------------------------

describe('Dietary Tag Edge Cases', () => {
  it('should handle recipes with empty tags array', () => {
    createTestRecipe({ title: 'No Tags Recipe', dietary_tags: '[]' });

    const results = filterByDietaryTags(getAllRecipes(), ['vegan']);
    expect(results).toHaveLength(0);
  });

  it('should handle recipes with default tags value', () => {
    // Insert directly to get the DEFAULT '[]' from schema
    db.prepare("INSERT INTO recipes (title, ingredients, instructions) VALUES (?, ?, ?)")
      .run('Default Tags', 'stuff', 'do it');

    const row = db.prepare('SELECT dietary_tags FROM recipes WHERE title = ?').get('Default Tags');
    expect(row.dietary_tags).toBe('[]');
    expect(JSON.parse(row.dietary_tags)).toEqual([]);
  });

  it('should filter with an empty required tags array (matches everything)', () => {
    createTestRecipe({ title: 'Any Recipe', dietary_tags: JSON.stringify(['vegan']) });
    createTestRecipe({ title: 'Another Recipe', dietary_tags: '[]' });

    const results = filterByDietaryTags(getAllRecipes(), []);
    expect(results).toHaveLength(2);
  });

  it('should handle a recipe with all possible dietary tags', () => {
    createTestRecipe({
      title: 'Everything Recipe',
      dietary_tags: JSON.stringify(DIETARY_LABELS),
    });

    // Should match any single tag
    DIETARY_LABELS.forEach(label => {
      const results = filterByDietaryTags(getAllRecipes(), [label]);
      expect(results).toHaveLength(1);
      expect(results[0].title).toBe('Everything Recipe');
    });
  });

  it('should not match invalid/unknown dietary tags', () => {
    createTestRecipe({
      title: 'Normal Recipe',
      dietary_tags: JSON.stringify(['vegetarian']),
    });

    const results = filterByDietaryTags(getAllRecipes(), ['halal']);
    expect(results).toHaveLength(0);
  });

  it('should correctly parse dietary_tags JSON from the database', () => {
    const tags = ['vegan', 'gluten-free', 'nut-free'];
    createTestRecipe({
      title: 'JSON Test',
      dietary_tags: JSON.stringify(tags),
    });

    const row = db.prepare('SELECT dietary_tags FROM recipes WHERE title = ?').get('JSON Test');
    const parsed = JSON.parse(row.dietary_tags);
    expect(parsed).toEqual(tags);
    expect(Array.isArray(parsed)).toBe(true);
  });
});
